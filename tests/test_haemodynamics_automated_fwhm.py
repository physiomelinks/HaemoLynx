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


# --- _clip_profile_to_central_lobe ------------------------------------------


def _noisy_gaussian_profile(
    *,
    true_diameter: float,
    noise_sigma: float,
    half_extent: float = 30.0,
    step: float = 0.2,
    peak: float = 200.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A single-lobe Gaussian profile (FWHM == true_diameter) with additive
    per-sample Gaussian noise -- the same shape a real, noisy transverse ray
    produces for a single, uncomplicated vessel crossing."""
    rng = np.random.default_rng(seed)
    x = np.arange(-half_extent, half_extent + step, step, dtype=float)
    sigma = true_diameter / 2.3548
    y = peak * np.exp(-(x**2) / (2.0 * sigma**2))
    y = y + rng.normal(0.0, noise_sigma, size=y.shape)
    return x, y


def test_clip_profile_to_central_lobe_survives_light_noise_on_a_clean_single_lobe():
    """Regression: the valley/rise decision used to run against raw,
    per-sample intensities. Gaussian noise as small as 2% of the peak
    amplitude was enough for an ordinary noisy uptick right after the peak
    to look like the start of a second lobe, truncating the accepted
    window to a sliver and collapsing the fitted diameter by 8x or more
    (confirmed independently on this exact fixture). The smoothed decision
    in the current code must keep the window close to the true lobe width
    despite that same noise."""
    true_diameter = 16.0
    x, y = _noisy_gaussian_profile(
        true_diameter=true_diameter, noise_sigma=0.02 * 200.0, seed=1
    )
    x_fit, y_fit = automated._clip_profile_to_central_lobe(x, y)
    window = float(x_fit.max() - x_fit.min())
    # A single, uncomplicated lobe should keep most of the sampled extent --
    # the old, unsmoothed decision collapsed this to under 2um.
    assert window > 3.0 * true_diameter

    d, _x0, _r2 = automated._fwhm_gaussian_fit_with_diagnostics(x_fit, y_fit)
    assert d is not None
    assert d == pytest.approx(true_diameter, rel=0.25)


@pytest.mark.parametrize("noise_sigma_fraction", [0.0, 0.01, 0.02, 0.05])
def test_clip_profile_to_central_lobe_diameter_estimate_is_stable_across_noise_levels(
    noise_sigma_fraction,
):
    """Companion to the regression above, sweeping several noise levels:
    the fitted diameter should stay in the right ballpark at every level
    tested, not just the specific one that first exposed the bug."""
    true_diameter = 16.0
    peak = 200.0
    x, y = _noisy_gaussian_profile(
        true_diameter=true_diameter,
        noise_sigma=noise_sigma_fraction * peak,
        peak=peak,
        seed=2,
    )
    x_fit, y_fit = automated._clip_profile_to_central_lobe(x, y)
    d, _x0, _r2 = automated._fwhm_gaussian_fit_with_diagnostics(x_fit, y_fit)
    assert d is not None
    assert d == pytest.approx(true_diameter, rel=0.3)


def test_clip_profile_to_central_lobe_still_truncates_a_genuine_second_lobe():
    """The noise-robustness fix must not turn the clip into a no-op: a
    profile with a real, well-separated second peak (a ray reaching a
    neighbouring branch) should still be cut back to the central lobe."""
    x = np.linspace(-30.0, 30.0, 301, dtype=float)
    sigma = 16.0 / 2.3548
    primary = 200.0 * np.exp(-(x**2) / (2.0 * sigma**2))
    second_lobe_center = 22.0
    secondary = 180.0 * np.exp(-((x - second_lobe_center) ** 2) / (2.0 * sigma**2))
    y = primary + secondary

    x_fit, _y_fit = automated._clip_profile_to_central_lobe(x, y)
    # The right side must be cut back before it reaches the second lobe;
    # the left side has no such feature in this fixture and is free to
    # keep its full sampled extent.
    assert x_fit.max() < second_lobe_center - sigma


# --- _local_same_edge_window ------------------------------------------------


def _window(diameter_estimate, **overrides):
    kwargs = dict(
        same_edge_arc_window_um=None,
        same_edge_arc_window_min_um=1.0,
        same_edge_arc_window_multiplier=1.0,
        sample_spacing_along_edge_um=5.0,
        transverse_profile_step_um=0.2,
    )
    kwargs.update(overrides)
    return automated._local_same_edge_window(diameter_estimate, **kwargs)


def test_local_same_edge_window_scales_with_the_diameter_estimate():
    """window = max(min, multiplier x diameter_estimate) -- the documented
    contract this function implements."""
    assert _window(20.0) == pytest.approx(20.0)
    assert _window(20.0, same_edge_arc_window_multiplier=2.0) == pytest.approx(40.0)


def test_local_same_edge_window_respects_its_own_floor():
    assert _window(0.5, same_edge_arc_window_min_um=3.0) == pytest.approx(3.0)


def test_local_same_edge_window_falls_back_to_spacing_only_with_no_estimate():
    """Regression: a genuinely unknown diameter (<= 0, the very first
    sampling pass when diameter_guess_um=None) falls back to sample
    spacing -- but only then, not whenever a real estimate is available."""
    assert _window(0.0, sample_spacing_along_edge_um=7.0) == pytest.approx(7.0)
    assert _window(-1.0, sample_spacing_along_edge_um=7.0) == pytest.approx(7.0)


def test_local_same_edge_window_an_explicit_override_wins():
    assert _window(20.0, same_edge_arc_window_um=2.5) == pytest.approx(2.5)


def test_local_same_edge_window_does_not_use_sample_spacing_when_an_estimate_exists():
    """Regression: the window must scale with the vessel's own width, not
    with an unrelated quantity like how far apart FWHM samples are placed
    along the edge -- a wide vessel measured with fine sample spacing used
    to get a window pinned to that spacing regardless of its own diameter,
    silently shrinking the transverse sampling window (and, via
    cap_half_extent_by_nonlocal_same_edge_distance, the visualized/measured
    extent) for any vessel wider than the chosen spacing."""
    window = _window(30.0, sample_spacing_along_edge_um=2.0)  # spacing << diameter
    assert window == pytest.approx(30.0)
    assert window > 2.0


# --- _local_max_center_offset ------------------------------------------------


def _offset(diameter_estimate, **overrides):
    kwargs = dict(max_fit_center_offset_um=1.5, max_fit_center_offset_fraction_of_diameter=0.3)
    kwargs.update(overrides)
    return automated._local_max_center_offset(diameter_estimate, **kwargs)


def test_local_max_center_offset_scales_with_the_diameter_estimate():
    assert _offset(20.0) == pytest.approx(6.0)
    assert _offset(20.0, max_fit_center_offset_fraction_of_diameter=0.5) == pytest.approx(10.0)


def test_local_max_center_offset_respects_its_own_floor():
    """A narrow vessel's own scaled offset is smaller than the fixed
    floor -- the floor wins, exactly like today's behaviour for a vessel
    at roughly the scale max_fit_center_offset_um was chosen for."""
    assert _offset(1.0) == pytest.approx(1.5)


def test_local_max_center_offset_with_no_diameter_estimate_uses_only_the_floor():
    """Regression: a genuinely unknown diameter (<= 0, before any fit
    exists) must not be treated as a diameter of 0 -- that would make the
    scaled term 0 and (if it somehow won) reject every sample outright."""
    assert _offset(0.0) == pytest.approx(1.5)
    assert _offset(-1.0) == pytest.approx(1.5)


def test_local_max_center_offset_does_not_use_a_fixed_value_for_a_wide_vessel():
    """Regression: max_fit_center_offset_um used to be the ONLY threshold,
    regardless of vessel width -- comfortably loose for a narrow capillary,
    disproportionately strict for a wide vessel. A 30um vessel's own
    scaled offset must exceed the fixed floor that was tuned for a much
    narrower one."""
    wide_vessel_offset = _offset(30.0)
    assert wide_vessel_offset == pytest.approx(9.0)
    assert wide_vessel_offset > 1.5


# --- _edge_diameter_guess -----------------------------------------------------


def test_edge_diameter_guess_prefers_the_edge_attribute_when_present():
    data = {"edt_diameter_um": 12.0}
    guess = automated._edge_diameter_guess(
        data, diameter_guess_edge_attribute="edt_diameter_um", fallback_diameter_guess=2.0,
    )
    assert guess == pytest.approx(12.0)


def test_edge_diameter_guess_falls_back_when_the_attribute_is_absent():
    guess = automated._edge_diameter_guess(
        {}, diameter_guess_edge_attribute="edt_diameter_um", fallback_diameter_guess=2.0,
    )
    assert guess == pytest.approx(2.0)


def test_edge_diameter_guess_falls_back_for_a_non_positive_or_non_finite_attribute():
    """Regression: a stray 0, negative, or NaN value under the attribute
    name must not silently become this edge's own diameter guess."""
    for bad_value in (0.0, -5.0, float("nan"), float("inf")):
        guess = automated._edge_diameter_guess(
            {"edt_diameter_um": bad_value},
            diameter_guess_edge_attribute="edt_diameter_um",
            fallback_diameter_guess=2.0,
        )
        assert guess == pytest.approx(2.0), f"bad_value={bad_value!r} should have fallen back"


def test_edge_diameter_guess_none_attribute_name_always_uses_the_fallback():
    """diameter_guess_edge_attribute=None reproduces this function's
    behaviour from before per-edge seeding existed: only the one global
    guess, even when the edge itself carries a real per-edge value."""
    guess = automated._edge_diameter_guess(
        {"edt_diameter_um": 12.0}, diameter_guess_edge_attribute=None, fallback_diameter_guess=2.0,
    )
    assert guess == pytest.approx(2.0)


def _cylinder_saturated_volume(
    nz: int, ny: int, nx: int, radius: float, saturation: float = 100.0, tau: float = 0.15
) -> np.ndarray:
    """A flat-topped (saturated) cylindrical cross-section: ~``saturation``
    inside ``radius`` with a short logistic falloff at the edge, ~0 outside
    -- mimics a detector-clipped signal (or a probability/segmentation
    field mistakenly used as raw intensity), not a real Gaussian bump.

    ``tau`` (the falloff's own width) needs to be sharp enough that it
    still reads as flat-topped once rasterised onto a 1-physical-unit-per-
    voxel grid and trilinearly re-interpolated for transverse sampling --
    interpolation smooths a logistic edge into something closer to a
    Gaussian ramp, and too soft a ``tau`` (this fixture's own original
    0.3) reads as under 0.85 once discretised even though the identical
    continuous profile clears it comfortably (see
    ``test_profile_plateau_shape_ratio_flags_flat_top_profile``, whose
    fixture is analytic, not voxelised, and does not have this problem).
    """
    zc, yc = (nz - 1) / 2.0, (ny - 1) / 2.0
    z = np.arange(nz, dtype=float)[:, None, None]
    y = np.arange(ny, dtype=float)[None, :, None]
    x = np.arange(nx, dtype=float)[None, None, :]
    r = np.sqrt((y - yc) ** 2 + (z - zc) ** 2) + 0.0 * (x - 10.0)
    return (saturation / (1.0 + np.exp((r - radius) / tau))).astype(np.float32)


def test_measure_edge_diameters_rejects_saturated_synthetic_vessel(tmp_path: Path):
    """A flat-topped/saturated profile is a real risk for the parametric
    Gaussian fit's own R² (a heavily-overparameterized fit can still score
    deceptively well against a rounded plateau) -- the plateau-shape gate
    is independent of that fit and should measurably reduce how much of
    this saturated fixture gets accepted, whether that means fewer per-edge
    samples or the edge failing outright.

    The edge runs 36um (well past ``nonlocal_same_edge_arc_separation_um``'s
    own 6um default) so ``cap_half_extent_by_nonlocal_same_edge_distance``
    does not truncate the transverse sampling window down near the
    vessel's own radius before the profile ever reaches background --
    a short test edge (this fixture's original 16um) triggers exactly that
    cap and starves every gate of a profile wide enough to judge at all.
    ``reject_samples_with_center_offset``/``reject_samples_with_low_fit_r2``
    are disabled so this isolates the plateau-shape gate itself, matching
    what this test is actually about -- left at their defaults, the
    Gaussian fit's own R² gate (not the plateau gate) rejects everything
    regardless of ``reject_samples_with_plateau_shape``, which is what
    made the "ungated" sanity check below fail before this fixture was
    recalibrated.
    """
    nz, ny, nx_dim = 15, 15, 41
    raw = _cylinder_saturated_volume(nz, ny, nx_dim, radius=3.0)
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(str(raw_path), raw)
    zc, yc = (nz - 1) / 2.0, (ny - 1) / 2.0

    def _build_graph() -> nx.MultiGraph:
        graph = nx.MultiGraph()
        graph.add_node(0, pos=np.array([zc, yc, 2.0], dtype=float))
        graph.add_node(1, pos=np.array([zc, yc, 38.0], dtype=float))
        voxels = [(zc, yc, float(x)) for x in range(2, 39)]
        graph.add_edge(0, 1, weight=1.0, length=36.0, branch_order="B01", voxels=voxels)
        return graph

    common = dict(
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=5.0,
        transverse_profile_step_um=0.2,
        transverse_half_extent_um=7.0,
        diameter_guess_um=2.0,
        reject_samples_with_center_offset=False,
        reject_samples_with_low_fit_r2=False,
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
    assert ungated_n > 0  # sanity: the fixture is fittable at all without the gate
    assert gated_n < ungated_n  # the plateau gate measurably rejects samples


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


def test_gram_schmidt_perpendicular_uses_the_least_parallel_axis():
    """Regression: the docstring promises the reference axis *least*
    parallel to the tangent is subtracted, not the most parallel one.

    ``t`` is mostly aligned with x (axis 2), moderately with y (axis 1), and
    least with z (axis 0). Subtracting the least-parallel axis (z, correct)
    leaves the result dominated by z; subtracting the most-parallel one
    (x, the inverted-condition bug) leaves it dominated by y instead.
    """
    t = np.array([0.1, 0.3, 0.95], dtype=float)
    n = automated._gram_schmidt_perpendicular(t)
    assert abs(n[0]) > abs(n[1])
    assert abs(n[0]) > abs(n[2])


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


def test_measure_edge_diameters_seeds_the_first_pass_from_an_edt_diameter_attribute(
    tmp_path: Path, monkeypatch
):
    """Regression: diameter_guess_um used to be the only seed for every
    edge's own first sampling pass, one number shared regardless of a
    given edge's real width. An edge already carrying edt_diameter_um
    (what haemodynamics.edt_diameter.measure_edge_diameters_from_binary_mask
    writes) must seed its own first pass from that instead of the one
    global fallback -- confirmed here by recording the half-extent each
    first sampling call actually requests.
    """
    nz, ny, nx_dim = 11, 11, 21
    raw = _cylinder_gaussian_volume(nz, ny, nx_dim, sigma=1.5)
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(str(raw_path), raw)

    def _build_graph(edt_value: float | None) -> nx.MultiGraph:
        G = nx.MultiGraph()
        G.add_node(0, pos=np.array([5.0, 5.0, 2.0], dtype=float))
        G.add_node(1, pos=np.array([5.0, 5.0, 18.0], dtype=float))
        voxels = [(5.0, 5.0, float(x)) for x in range(2, 19)]
        edge_kwargs = {} if edt_value is None else {"edt_diameter_um": edt_value}
        G.add_edge(0, 1, weight=1.0, length=16.0, branch_order="B01", voxels=voxels, **edge_kwargs)
        return G

    recorded_half_extents: list[float] = []
    real_sample = automated._sample_transverse_profile

    def recording_sample(raw_arr, labels, center, tangent, assigned, half_extent, *args, **kwargs):
        recorded_half_extents.append(half_extent)
        return real_sample(raw_arr, labels, center, tangent, assigned, half_extent, *args, **kwargs)

    common = dict(
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=20.0,  # > edge length: exactly one sample
        transverse_profile_step_um=0.2,
        transverse_half_extent_um=1.0,
        diameter_guess_um=2.0,  # the one global fallback
        min_total_extent_multiplier=3.0,
        cap_half_extent_by_nonlocal_same_edge_distance=False,
    )

    monkeypatch.setattr(automated, "_sample_transverse_profile", recording_sample)

    recorded_half_extents.clear()
    automated.measure_edge_diameters_fwhm_from_raw_tiff(_build_graph(None), **common)
    without_edt_first_half_extent = recorded_half_extents[0]

    recorded_half_extents.clear()
    automated.measure_edge_diameters_fwhm_from_raw_tiff(_build_graph(9.0), **common)
    with_edt_first_half_extent = recorded_half_extents[0]

    assert without_edt_first_half_extent == pytest.approx(max(1.0, 0.5 * 3.0 * 2.0))
    assert with_edt_first_half_extent == pytest.approx(max(1.0, 0.5 * 3.0 * 9.0))
    assert with_edt_first_half_extent > without_edt_first_half_extent


def _wavy_gaussian_volume(nz, ny, nx, radius, amplitude, wavelength, saturation=200.0):
    """A single vessel of ``radius`` whose centerline wiggles sinusoidally in
    y as a function of x -- mimics real vessel tortuosity, unlike every
    other fixture in this file, which is a straight tube."""
    zc = (nz - 1) / 2.0
    yc = ny / 2.0
    sigma = (2.0 * radius) / 2.3548  # so the fitted FWHM should recover 2*radius
    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    z = np.arange(nz, dtype=float)[:, None]
    y = np.arange(ny, dtype=float)[None, :]
    centerline_y = []
    for xi in range(nx):
        y_here = yc + amplitude * np.sin(2 * np.pi * xi / wavelength)
        centerline_y.append(y_here)
        r = np.sqrt((y - y_here) ** 2 + (z - zc) ** 2)
        vol[:, :, xi] = saturation * np.exp(-(r**2) / (2 * sigma**2))
    return vol, zc, np.array(centerline_y)


def test_transverse_window_scales_with_diameter_on_a_tortuous_vessel(tmp_path: Path):
    """Regression: for a vessel wide enough that its own diameter exceeds
    sample_spacing_along_edge_um, the same-edge-locality guards used to
    size their "how close counts as a genuine self-crossing" threshold off
    that unrelated sample spacing instead of the vessel's own width --
    correct for a ruler-straight vessel (where the guards never bind
    tightly enough to matter) but not for a realistically tortuous one.
    That specific diameter-scaling contract is pinned precisely and
    directly by the ``test_local_same_edge_window_*`` tests above, against
    the pure function itself; this integration test is a looser sanity
    floor on top, checking the whole pipeline still produces a usefully
    wide window end-to-end on a genuinely tortuous vessel.

    The exact ratio here also depends on ``_clip_profile_to_central_lobe``'s
    own central-lobe detection, which on this fixture's tight sinusoidal
    tortuosity (wavelength comparable to the vessel's own diameter) can
    correctly clip a wide pass's window down when it reaches a neighbouring
    loop of the same vessel -- independently of same-edge-locality sizing,
    and by design (see that function's docstring and
    ``_CLIP_DECISION_SMOOTHING_WINDOW_UM``: a per-sample-noise-robust
    decision was required after real (noisy) data showed the previous,
    unsmoothed valley/rise check collapsing a true profile's accepted
    window by 8x or more). Measured on this exact fixture, that fix moves
    the median ratio from ~1.99 to ~1.52 while leaving the actual fitted
    diameter itself essentially unchanged (~11.8um either way) -- i.e. the
    same-edge-locality contract is intact and the pipeline output is not
    materially different, only this debug-visualization ratio is smaller.
    The threshold below keeps margin under that, not pinned tightly to it.
    """
    true_diameter = 20.0
    nz, ny, nx_dim = 60, 100, 81
    raw, zc, centerline_y = _wavy_gaussian_volume(
        nz, ny, nx_dim, radius=true_diameter / 2.0, amplitude=10.0, wavelength=30.0
    )
    raw_path = tmp_path / "wavy.tif"
    tifffile.imwrite(str(raw_path), raw)

    xs = list(range(5, 76))
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([zc, centerline_y[5], 5.0], dtype=float))
    G.add_node(1, pos=np.array([zc, centerline_y[75], 75.0], dtype=float))
    voxels = [(zc, float(centerline_y[x]), float(x)) for x in xs]
    G.add_edge(0, 1, weight=1.0, length=70.0, branch_order="B01", voxels=voxels)

    summary = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=5.0,  # well under the vessel's own 20um diameter
        transverse_profile_step_um=0.2,
        transverse_half_extent_um=3.0,
        diameter_guess_um=true_diameter,
        store_profile_debug=True,
    )
    assert summary["edges_measured"] == 1
    data = G.edges[0, 1, 0]
    lines = data["fwhm_profile_lines_phys"]
    assert lines, "expected at least one accepted transverse profile line"
    ratios = [float(np.linalg.norm(line[-1] - line[0])) / true_diameter for line in lines]
    # Comfortable margin below this fixture's current measured value
    # (~1.52, see the docstring above) -- a sanity floor that the window
    # is still usefully wide end-to-end, not a tight pin on the exact
    # number, which also depends on _clip_profile_to_central_lobe's own
    # (independently, directly tested) central-lobe detection.
    assert float(np.median(ratios)) > 1.3


def test_center_offset_gate_scales_with_diameter_estimate(tmp_path: Path):
    """Regression: max_fit_center_offset_um used to be the only threshold
    for how far a fitted center may sit from the sampling origin,
    regardless of vessel width. A wide (30um) vessel whose recorded
    centerline is only mildly (2um) off from its own true center -- a
    modest imprecision any real skeleton/graph can reasonably have --
    used to fail every sample outright under the fixed 1.5um floor alone,
    even though a 2um offset on a 30um vessel is a trivial, unremarkable
    fraction of its own radius. Confirms the offset gate, scaled to the
    diameter estimate, now accepts it.
    """
    true_diameter = 30.0
    sigma = true_diameter / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    nz, ny, nx_dim = 120, 120, 21
    zc, yc = 60.0, 60.0
    z = np.arange(nz, dtype=float)[:, None, None]
    y = np.arange(ny, dtype=float)[None, :, None]
    x = np.arange(nx_dim, dtype=float)[None, None, :]
    r2 = (y - yc) ** 2 + (z - zc) ** 2 + 0.0 * (x - 10.0)
    raw = (100.0 * np.exp(-r2 / (2.0 * sigma**2))).astype(np.float32)
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(str(raw_path), raw)

    centerline_offset = 2.0  # the recorded centerline is 2um off the true peak
    G = nx.MultiGraph()
    G.add_node(0, pos=np.array([zc, yc + centerline_offset, 2.0], dtype=float))
    G.add_node(1, pos=np.array([zc, yc + centerline_offset, 18.0], dtype=float))
    voxels = [(zc, yc + centerline_offset, float(x)) for x in range(2, 19)]
    G.add_edge(0, 1, weight=1.0, length=16.0, branch_order="B01", voxels=voxels)

    common = dict(
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=20.0,  # > edge length: exactly one sample
        transverse_profile_step_um=0.2,
        transverse_half_extent_um=8.0,
        diameter_guess_um=true_diameter,
    )
    fixed_only = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G, max_fit_center_offset_fraction_of_diameter=0.0, **common,
    )
    assert fixed_only["edges_measured"] == 0, (
        "a 2um offset should still exceed the fixed 1.5um floor alone "
        "(fraction_of_diameter=0.0 reproduces that old, fixed-only behaviour)"
    )

    scaled = automated.measure_edge_diameters_fwhm_from_raw_tiff(G, **common)
    assert scaled["edges_measured"] == 1
    d = scaled["per_edge"][0]["fwhm_diameter_um"]
    assert abs(d - true_diameter) < 2.0


def test_transverse_widening_loop_converges_beyond_two_widening_passes(
    tmp_path: Path, monkeypatch
):
    """Regression: the widening cascade used to be hand-unrolled to
    exactly 3 passes (the initial one, then two widenings) regardless of
    whether that was enough to converge. A slowly-converging sequence of
    fitted diameters needs 6 widenings to fully converge -- confirms the
    loop now keeps going as far as max_transverse_widen_passes allows,
    and that the parameter genuinely caps it rather than being ignored.
    """
    nz, ny, nx_dim = 11, 11, 21
    raw = _cylinder_gaussian_volume(nz, ny, nx_dim, sigma=1.5)
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(str(raw_path), raw)

    def _build_graph() -> nx.MultiGraph:
        G = nx.MultiGraph()
        G.add_node(0, pos=np.array([5.0, 5.0, 2.0], dtype=float))
        G.add_node(1, pos=np.array([5.0, 5.0, 18.0], dtype=float))
        voxels = [(5.0, 5.0, float(x)) for x in range(2, 19)]
        G.add_edge(0, 1, weight=1.0, length=16.0, branch_order="B01", voxels=voxels)
        return G

    # Each successive "fit" asks for a bigger window than the last pass
    # granted, right up until the final two calls agree -- needs 6
    # widenings (7 total passes) to settle.
    sequence = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 12.0]
    calls = {"n": 0}

    def fake_fit(pos_fit, prof_fit, **kwargs):
        idx = min(calls["n"], len(sequence) - 1)
        calls["n"] += 1
        return sequence[idx], 0.0, 1.0

    monkeypatch.setattr(automated, "_fwhm_gaussian_fit_with_diagnostics", fake_fit)

    common = dict(
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=20.0,  # > edge length: exactly one sample
        transverse_profile_step_um=0.2,
        transverse_half_extent_um=1.0,
        diameter_guess_um=1.0,
        min_total_extent_multiplier=3.0,
        cap_half_extent_by_nonlocal_same_edge_distance=False,
        reject_samples_with_center_offset=False,
        reject_samples_with_low_fit_r2=False,
        reject_samples_with_plateau_shape=False,
    )

    calls["n"] = 0
    capped = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        _build_graph(), max_transverse_widen_passes=1, **common,
    )
    # Only 2 total passes allowed: settles on the second call's own value,
    # nowhere near the sequence's final, fully-converged one.
    assert capped["per_edge"][0]["fwhm_diameter_um"] == pytest.approx(4.0)

    calls["n"] = 0
    converged = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        _build_graph(), max_transverse_widen_passes=10, **common,
    )
    assert converged["per_edge"][0]["fwhm_diameter_um"] == pytest.approx(sequence[-1])


def test_transverse_widening_keeps_the_last_good_measurement_if_a_wider_pass_fails(
    tmp_path: Path, monkeypatch
):
    """Regression: if the initial pass's fit succeeded but a subsequent,
    wider pass's own fit failed a rejection gate, the whole sample used to
    be discarded outright -- even though the narrower, already-accepted
    measurement was perfectly good. Widening is only ever an attempt to
    improve an already-accepted measurement; it must never cost the
    sample the measurement it already had.
    """
    nz, ny, nx_dim = 11, 11, 21
    raw = _cylinder_gaussian_volume(nz, ny, nx_dim, sigma=1.5)
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(str(raw_path), raw)

    def _build_graph() -> nx.MultiGraph:
        G = nx.MultiGraph()
        G.add_node(0, pos=np.array([5.0, 5.0, 2.0], dtype=float))
        G.add_node(1, pos=np.array([5.0, 5.0, 18.0], dtype=float))
        voxels = [(5.0, 5.0, float(x)) for x in range(2, 19)]
        G.add_edge(0, 1, weight=1.0, length=16.0, branch_order="B01", voxels=voxels)
        return G

    calls = {"n": 0}

    def fake_fit(pos_fit, prof_fit, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return 5.0, 0.0, 1.0  # accepted
        return None, None, None  # every wider pass's own fit fails

    monkeypatch.setattr(automated, "_fwhm_gaussian_fit_with_diagnostics", fake_fit)

    summary = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        _build_graph(),
        raw_tiff_path=raw_path,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        sample_spacing_along_edge_um=20.0,  # > edge length: exactly one sample
        transverse_profile_step_um=0.2,
        transverse_half_extent_um=1.0,
        diameter_guess_um=1.0,
        min_total_extent_multiplier=3.0,
        cap_half_extent_by_nonlocal_same_edge_distance=False,
        reject_samples_with_center_offset=False,
        reject_samples_with_low_fit_r2=False,
        reject_samples_with_plateau_shape=False,
        max_transverse_widen_passes=5,
    )
    assert summary["edges_measured"] == 1
    assert summary["per_edge"][0]["fwhm_diameter_um"] == pytest.approx(5.0)
    assert calls["n"] > 1, "the widening pass that fails must actually have been attempted"


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


def _parallel_edges_sharing_one_wide_vessel(
    tmp_path: Path, *, true_diameter: float, separation: float
) -> tuple[Path, nx.MultiGraph, tuple[float, float, float]]:
    """Two straight, parallel graph edges *separation* um apart, both embedded
    in ONE wide vessel body centred on the midpoint between them.

    Models a real, recurring skeleton/graph-build outcome for a fat vessel:
    a spurious double centerline (e.g. from a near-loop the thickness-gated
    skeletoniser did not fully collapse), or two genuinely separate branches
    that happen to run close together through a shared, wide trunk. Either
    way, each edge's own transverse ray finds the *other* edge's centerline
    voxel well inside the true vessel radius when *separation* is small
    relative to *true_diameter*.
    """
    sigma = true_diameter / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    voxel = (0.25, 0.25, 0.25)
    z0, x0, x1 = 10.0, 5.0, 35.0
    y_a = 15.0
    y_b = y_a + separation
    mid_y = (y_a + y_b) / 2.0

    nz, ny, nx_dim = 80, 140, 160
    z = np.arange(nz, dtype=float)[:, None, None] * voxel[0]
    y = np.arange(ny, dtype=float)[None, :, None] * voxel[1]
    x = np.arange(nx_dim, dtype=float)[None, None, :] * voxel[2]
    t = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    d2 = (z - z0) ** 2 + (y - mid_y) ** 2 + 0.0 * t
    raw = (100.0 * np.exp(-d2 / (2.0 * sigma**2))).astype(np.float32)
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(str(raw_path), raw)

    G = nx.MultiGraph()
    n_points = 400  # dense enough that no voxel along either centerline is missed
    tvals = np.linspace(0.0, 1.0, n_points)
    voxels_a = [(z0, y_a, x0 + t_ * (x1 - x0)) for t_ in tvals]
    voxels_b = [(z0, y_b, x0 + t_ * (x1 - x0)) for t_ in tvals]
    G.add_node(0, pos=np.array([z0, y_a, x0]))
    G.add_node(1, pos=np.array([z0, y_a, x1]))
    G.add_node(2, pos=np.array([z0, y_b, x0]))
    G.add_node(3, pos=np.array([z0, y_b, x1]))
    G.add_edge(0, 1, weight=1.0, length=x1 - x0, branch_order="B01", voxels=voxels_a)
    G.add_edge(2, 3, weight=1.0, length=x1 - x0, branch_order="B02", voxels=voxels_b)
    return raw_path, G, voxel


def test_allow_crossing_other_edges_recovers_a_wide_vessel_split_into_close_edges(
    tmp_path: Path,
):
    """Regression: a transverse ray used to hard-stop the instant it reached
    *any other edge's own* centerline voxel, even one just a few microns
    away -- a poor proxy for "this is a different vessel" when that nearby
    edge is really another fragment of the same wide vessel (e.g. a
    spurious double centerline a thickness-gated skeletoniser did not fully
    collapse on a fat vessel). Two edges 6um apart, both embedded in one
    true 16um-diameter vessel, must both recover that true diameter under
    the default settings -- confirmed to fail before this fix by stashing
    ``automated.py`` (see the module docstring for the general convention).
    """
    true_diameter = 16.0
    raw_path, G, voxel = _parallel_edges_sharing_one_wide_vessel(
        tmp_path, true_diameter=true_diameter, separation=6.0
    )
    summary = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=voxel,
        sample_spacing_along_edge_um=2.0,
        transverse_profile_step_um=0.2,
        transverse_half_extent_um=20.0,
        min_total_extent_multiplier=3.0,
    )
    assert summary["edges_measured"] == 2
    for entry in summary["per_edge"]:
        assert abs(entry["fwhm_diameter_um"] - true_diameter) < 0.5

    # allow_crossing_other_edges=False restores the stricter, topology-only
    # stop for anyone who prefers it -- explicitly asking for it must still
    # work as its own documented, opt-out knob.
    restored = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=voxel,
        sample_spacing_along_edge_um=2.0,
        transverse_profile_step_um=0.2,
        transverse_half_extent_um=20.0,
        min_total_extent_multiplier=3.0,
        allow_crossing_other_edges=False,
    )
    assert restored["edges_measured"] + len(restored["edges_skipped"]) == 2


def test_center_offset_gate_uses_the_just_fitted_diameter_not_a_stale_guess(
    tmp_path: Path,
):
    """Regression: the center-offset gate scaled its tolerance to the
    pre-fit diameter *estimate* only -- on an edge with no per-edge
    diameter guess, that estimate starts small (or absent), so an
    excellent, well-fit-but-off-center first pass on a genuinely wide
    vessel was rejected purely because the offset looked large relative to
    a stale, too-small guess, and the widening loop that would otherwise
    grow the estimate never got a chance to run (a rejected first pass
    aborts immediately). Scaling against max(guess, this pass's own fitted
    diameter) fixes it without loosening the gate for a genuinely oblique
    sample -- confirmed here with true_diameter=16, no diameter_guess_um,
    and a real 3um centerline offset (a plausible, modest skeleton
    imprecision on a vessel this wide).
    """
    true_diameter = 16.0
    raw_path, G, voxel = _parallel_edges_sharing_one_wide_vessel(
        tmp_path, true_diameter=true_diameter, separation=6.0
    )
    # Edge (0, 1)'s own centerline is 3um off the true vessel peak (at the
    # midpoint between it and the unrelated edge (2, 3)) -- drop the other
    # edge so this test isolates the offset-gate fix alone.
    G.remove_edge(2, 3)
    G.remove_nodes_from([2, 3])

    summary = automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G,
        raw_tiff_path=raw_path,
        voxel_size_zyx=voxel,
        sample_spacing_along_edge_um=2.0,
        transverse_profile_step_um=0.2,
        transverse_half_extent_um=20.0,
        diameter_guess_um=None,
        min_total_extent_multiplier=3.0,
    )
    assert summary["edges_measured"] == 1
    assert abs(summary["per_edge"][0]["fwhm_diameter_um"] - true_diameter) < 0.5
