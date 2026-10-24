"""Vessels of known lumen planted in an endothelial image
(haemolynx.haemodynamics.endothelial_planted), which the endothelial settings
optimiser scores accuracy by: the ring fit reads them at their true lumen,
like the real vessel beside them, at any voxel size, wall and blur.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from haemolynx.haemodynamics import endothelial, endothelial_planted
from haemolynx.haemodynamics.endothelial import measure_edge_diameters_from_endothelium
from haemolynx.haemodynamics.fwhm_diameter import _arc_length_parameterize

from endothelial_fixtures import hollow_vessel_scene

#: One vessel through the middle of the volume, tilted out of every axis,
#: with room beside it to plant in.
ONE_VESSEL = (((0.15, 0.25, 1.0), (0.5, 0.5, 0.5), 18.0),)

#: (voxel size zyx, optics zyx, wall): a two-photon stack's anisotropic
#: voxels, a fine isotropic one, a coarse one -- none of them E14.5's.
IMAGES = [
    ((1.0, 0.98, 0.98), (2.0, 0.5, 0.5), 1.0),
    ((2.0, 0.5, 0.5), (2.5, 0.4, 0.4), 0.8),
    ((0.6, 0.6, 0.6), (1.2, 0.6, 0.6), 1.5),
    ((1.5, 1.2, 1.2), (2.0, 0.8, 0.8), 1.2),
]


def _scene(voxel, optics, wall, lumen=7.0):
    scene = hollow_vessel_scene(
        voxel_size_zyx=voxel, optics_um=optics, wall_um=wall, lumen_um=lumen,
        extent_um=(44.0, 96.0, 64.0), vessels=ONE_VESSEL, pieces=2, seed=3,
    )
    for _u, _v, data in scene.graph.edges(data=True):
        data["edt_diameter_um"] = lumen + 2 * wall  # a mask of the stained vessel
    return scene


def _planted(scene, sample_size=4):
    return endothelial_planted.plant_rings(
        scene.volume, scene.graph, list(scene.graph.edges(keys=True)), scene.mask,
        scene.voxel_size_zyx, psf_sigma_zyx=scene.psf_seen_zyx, wall_um=scene.wall_um,
        rng=np.random.default_rng(1), sample_size=sample_size,
    )


def _read(probe, volume, scene):
    measure_edge_diameters_from_endothelium(
        probe, endothelial_volume=volume, voxel_size_zyx=scene.voxel_size_zyx,
        psf_sigma_zyx=scene.psf_seen_zyx, wall_um=scene.wall_um, branch_endpoint_exclusion_um=0.0,
    )


@pytest.mark.parametrize("voxel, optics, wall", IMAGES, ids=["claudin", "anisotropic", "fine", "coarse"])
def test_the_ring_fit_reads_a_planted_vessel_at_its_true_lumen(voxel, optics, wall):
    scene = _scene(voxel, optics, wall)

    planted = _planted(scene)

    assert planted is not None and planted.probe.number_of_edges() >= 2
    assert planted.reference_rings >= 1
    probe = planted.probe.copy()
    _read(probe, planted.volume, scene)
    for _u, _v, data in probe.edges(data=True):
        true = data["planted_diameter_um"]
        assert 0.75 * 0.9 * scene.lumen_um <= true <= 1.25 * 1.1 * scene.lumen_um
        assert data["endothelial_status"] == "measured", data["endothelial_status"]
        assert data["endothelial_diameter_um"] == pytest.approx(true, rel=0.06)
    report = endothelial_planted.planted_lumen_report(probe)
    assert report["measured_fraction"] == 1.0 and report["relative_error"] < 0.06


@pytest.mark.parametrize("voxel, optics, wall", IMAGES[:2], ids=["claudin", "anisotropic"])
def test_a_planted_vessel_is_as_bright_as_the_ring_it_is_drawn_like(voxel, optics, wall):
    """Its wall is drawn at the level the real vessel's ring fits at, so a
    gate on the ring's contrast sees the image's own rings."""
    scene = _scene(voxel, optics, wall)
    planted = _planted(scene, sample_size=2)
    real = next(iter(scene.graph.edges(keys=True)))

    def wall_level(volume, data):
        poly = np.asarray(data["voxels"], dtype=float)
        s, total = _arc_length_parameterize(poly)
        fit, _why = endothelial.fit_ring_section(
            volume, poly, s, total, 0.5 * total, scene.voxel_size_zyx,
            guide_um=float(data["edt_diameter_um"]), psf_sigma_zyx=scene.psf_seen_zyx,
            wall_um=scene.wall_um,
        )
        return fit.wall_level

    reference = wall_level(scene.volume, scene.graph.edges[real])
    for _u, _v, data in planted.probe.edges(data=True):
        assert wall_level(planted.volume, data) == pytest.approx(reference, rel=0.15)


def test_planted_vessels_keep_clear_of_the_vessels_and_the_faces():
    scene = _scene(*IMAGES[0])
    planted = _planted(scene, sample_size=6)
    spacing = np.asarray(scene.voxel_size_zyx)
    to_vessel = distance_transform_edt(~scene.mask, sampling=spacing)
    far_face = (np.asarray(scene.mask.shape) - 1) * spacing
    optics = planted.optics_psf_um

    for _u, _v, data in planted.probe.edges(data=True):
        line = np.asarray(data["voxels"])
        outer = data["planted_diameter_um"] + 2 * planted.wall_um
        index = tuple(np.rint(line / spacing).astype(int).T)
        clearance = endothelial_planted.PLANTED_CLEARANCE_MULTIPLE * outer + 2 * max(optics)
        assert to_vessel[index].min() >= clearance - spacing.max()
        margin = 0.5 * outer + 3 * max(planted.psf_sigma_zyx) + 1.0
        assert np.all(line >= margin) and np.all(line <= far_face - margin)
    # The real vessel is untouched where it is.
    assert np.allclose(planted.volume[scene.mask], scene.volume[scene.mask], atol=1.0)


def test_nothing_is_planted_where_no_vessel_shows_a_ring():
    scene = _scene(*IMAGES[0])
    flat = np.full_like(scene.volume, 5.0)

    assert endothelial_planted.plant_rings(
        flat, scene.graph, list(scene.graph.edges(keys=True)), scene.mask, scene.voxel_size_zyx,
        psf_sigma_zyx=scene.psf_seen_zyx, wall_um=scene.wall_um, rng=np.random.default_rng(0),
    ) is None


def test_the_report_counts_an_unread_planted_vessel_as_all_wrong():
    probe = nx.MultiGraph()
    probe.add_edge(0, 1, key=0, planted_diameter_um=8.0, endothelial_status="measured",
                   endothelial_diameter_um=9.0)
    probe.add_edge(2, 3, key=0, planted_diameter_um=6.0, endothelial_status="failed:low_contrast")
    probe.add_edge(4, 5, key=0, planted_diameter_um=4.0, endothelial_status="measured",
                   endothelial_diameter_um=12.0)

    report = endothelial_planted.planted_lumen_report(probe)

    assert report["planted"] == 3 and report["planted_measured"] == 2
    assert report["relative_error"] == pytest.approx((0.125 + 1.0 + 1.0) / 3)
    assert report["relative_error_measured"] == pytest.approx((0.125 + 1.0) / 2)
