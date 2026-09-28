"""The lumen-width methods side by side on vessels of known size -- a benchmark
that fails when one gets worse.

Plasma vessels (a filled lumen) for FWHM and the raw cross-section, endothelial
vessels (a hollow wall) for the ring fit, each at 3-10 um and running along x,
obliquely and along z, through a blur three to four times longer in z than
across, with shot noise. The tolerances are what each method achieves today:

* the raw cross-section: within 7% from 5 um and 10% at 3 um, at every
  orientation;
* the endothelial ring: within 6% from 4 um;
* FWHM: within 15% at every orientation -- a line profile reads a vessel
  lying in the image plane narrow, since the z-blur dims its edges; it once
  measured no z-running vessel at all.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.haemodynamics import automated
from haemolynx.haemodynamics.endothelial import measure_edge_diameters_from_endothelium
from haemolynx.haemodynamics.raw_section import measure_edge_diameters_from_raw_sections
from test_endothelial_diameter import PSF as RING_PSF, WALL, _ring  # noqa: E402
from test_raw_section_diameter import EFFECTIVE_PSF, VOXEL, _graph, _render  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.integration]

DIRECTIONS = {"along_x": (0, 0, 1), "oblique": (1, 1, 1), "along_z": (1, 0.05, 0.05)}


def _plasma_vessel(diameter, direction, seed):
    raw, mask, lines = _render([(diameter, direction, (0, 0, 0))], seed=seed, peak_counts=6.0)
    return raw, mask, _graph(lines, [diameter])


@pytest.mark.parametrize("diameter", [3.0, 5.0, 8.0])
@pytest.mark.parametrize("direction", list(DIRECTIONS), ids=list(DIRECTIONS))
def test_the_raw_cross_section_is_within_seven_percent_from_five_microns(diameter, direction):
    raw, mask, graph = _plasma_vessel(diameter, DIRECTIONS[direction], seed=int(10 * diameter))

    measure_edge_diameters_from_raw_sections(
        graph, raw_volume=raw, voxel_size_zyx=VOXEL, vessel_mask=mask,
        psf_sigma_zyx=EFFECTIVE_PSF, branch_endpoint_exclusion_um=0.0,
    )

    assert graph[0][1][0]["raw_section_diameter_um"] == pytest.approx(
        diameter, rel=0.10 if diameter < 5.0 else 0.07
    )


@pytest.mark.parametrize("diameter", [3.0, 5.0, 8.0])
@pytest.mark.parametrize("direction", list(DIRECTIONS), ids=list(DIRECTIONS))
def test_fwhm_is_within_fifteen_percent_at_every_orientation(diameter, direction):
    raw, mask, graph = _plasma_vessel(diameter, DIRECTIONS[direction], seed=int(10 * diameter) + 1)

    automated.measure_edge_diameters_fwhm_from_raw_tiff(
        graph, raw_tiff_path="unused", raw_volume=raw, vessel_mask=mask, voxel_size_zyx=VOXEL,
        sample_spacing_along_edge_um=2.0, transverse_profile_step_um=0.25,
        transverse_half_extent_um=6.0, min_accepted_samples=2,
    )

    assert graph[0][1][0]["fwhm_diameter_um"] == pytest.approx(diameter, rel=0.15)


@pytest.mark.parametrize("inner", [4.0, 6.0, 10.0])
@pytest.mark.parametrize("direction", list(DIRECTIONS), ids=list(DIRECTIONS))
def test_the_endothelial_ring_is_within_six_percent_from_four_microns(inner, direction):
    volume, line = _ring(inner, DIRECTIONS[direction], seed=int(10 * inner) + 2, shape=(40, 50, 50))
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, key=0, voxels=[tuple(p) for p in line], length=24.0, edt_diameter_um=0.7 * inner)

    measure_edge_diameters_from_endothelium(
        graph, endothelial_volume=volume, voxel_size_zyx=VOXEL, psf_sigma_zyx=RING_PSF,
        wall_um=WALL, branch_endpoint_exclusion_um=0.0,
    )

    assert graph[0][1][0]["endothelial_diameter_um"] == pytest.approx(inner, rel=0.06)
