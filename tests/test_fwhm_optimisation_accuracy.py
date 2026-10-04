"""Acceptance: does the FWHM optimiser steer towards widths that are right?

Real vessels of known width (the diameter benchmark's, see
``test_raw_section_diameter``): five that do not cross, 3-8 um, along x,
obliquely and along z. The optimiser never sees their true widths -- only
vessels it plants beside them (``haemodynamics.fwhm_planted``) -- so what is
tested is whether getting the planted ones right gets the real ones right.

Measured on this scene, the schema's defaults read the real vessels 8.9% off
on average. From a start poor in the ways that decide accuracy (a Gaussian
profile, no averaging along the vessel, a 2 um half-extent, one accepted
sample) at 22%, one pass of the search brought them to 8.4%. Per vessel it
does not reach the benchmark's own 15% everywhere -- neither do the defaults
on this scene (the 3 um vessel reads 23% narrow with them) -- so this asks
for what the planted vessels can promise: most of a poor start's error
removed, and a good start not made worse.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from haemolynx.haemodynamics import automated
from haemolynx.optimisation import fwhm_search as s
from haemolynx.pipeline.schema import SCHEMA
from test_raw_section_diameter import EFFECTIVE_PSF, VOXEL, _graph, _render  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.integration]

TUBES = [
    (3.0, (0, 0, 1), (0, -42, 0)),
    (4.0, (0, 0, 1), (10, -18, 0)),
    (5.0, (1, 0, 1), (0, 8, 0)),
    (8.0, (1, 0.05, 0.05), (0, 34, -32)),
    (6.0, (0, 0, 1), (-12, 50, 20)),
]
TRUTH = np.array([d for d, _direction, _shift in TUBES])


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    raw, mask, lines = _render(TUBES, shape=(60, 128, 128), peak_counts=6.0, seed=2)
    raw_path = tmp_path_factory.mktemp("accuracy") / "raw.tif"
    tifffile.imwrite(str(raw_path), raw)
    return raw, mask, lines, raw_path


def _defaults() -> dict:
    values = {field.name: field.default for field in SCHEMA if field.name.startswith("fwhm_")}
    values.update(
        raw_section_psf_sigma_xy_um=EFFECTIVE_PSF[1],
        raw_section_psf_sigma_z_um=EFFECTIVE_PSF[0],
        edt_fwhm_disagreement_warn_ratio=1.5,
    )
    return values


def _real_error(scene, settings) -> float:
    """The real vessels' mean relative width error with *settings*, an
    unmeasured one counting as 1."""
    raw, _mask, lines, _path = scene
    G = _graph(lines, list(TRUTH * 1.1))
    blurred_lumen = settings["fwhm_profile_model"] == "blurred_lumen"
    automated.measure_edge_diameters_fwhm_from_raw_tiff(
        G, raw_volume=raw, voxel_size_zyx=VOXEL, store_profile_debug=False,
        profile_psf_sigma_zyx=EFFECTIVE_PSF if blurred_lumen else None,
        **s._measurement_kwargs(settings),
    )
    widths = np.array([d.get("fwhm_diameter_um") or np.nan for _u, _v, d in G.edges(data=True)])
    errors = np.where(np.isfinite(widths), np.minimum(np.abs(widths - TRUTH) / TRUTH, 1.0), 1.0)
    return float(errors.mean())


def _optimised(scene, start: dict) -> dict:
    _raw, mask, lines, raw_path = scene
    result = s.optimise_fwhm_settings(
        _graph(lines, list(TRUTH * 1.1)), raw_tiff_path=raw_path, voxel_size_zyx=VOXEL,
        starting_values=start, sample_edge_count=10, vessel_mask=mask,
    )
    assert any(row.name == "planted vessels' width error" for row in result.scorecard)
    return {**start, **result.settings}


def test_the_search_removes_most_of_a_poor_starts_width_error(scene):
    poor = dict(
        _defaults(),
        fwhm_profile_model="gaussian",
        fwhm_longitudinal_average_um=0.0,
        fwhm_transverse_half_extent_um=2.0,
        fwhm_min_accepted_samples=1,
    )
    before = _real_error(scene, poor)

    after = _real_error(scene, _optimised(scene, poor))

    assert after <= 0.5 * before, (before, after)
    assert after <= _real_error(scene, _defaults()) + 0.02, (before, after)


def test_the_search_does_not_make_a_good_start_worse(scene):
    before = _real_error(scene, _defaults())

    after = _real_error(scene, _optimised(scene, _defaults()))

    assert after <= before + 0.01, (before, after)
