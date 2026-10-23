"""How a run resolves ``raw_image_stain="auto"`` (``pipeline/stages.py``'s
``resolve_raw_image_stain``) and hands the raw image on: the build's answer
is taken without opening the file; when the stage must read, the raw image
is cut to the network's crop box first and the volume goes to the
measurement, so the file is loaded once.
"""
from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from scipy import ndimage

from haemolynx.io.crop import GRAPH_ATTRIBUTE as CROP_ATTRIBUTE, CropBox
from haemolynx.pipeline import stages
from haemolynx.preprocessing.raw_stain import ENDOTHELIAL, PLASMA_COLUMN

VS = (1.0, 0.5, 0.5)
SHAPE = (48, 200, 200)
#: The box a cropped run was built in: y from voxel 40, where a raw image
#: read uncut puts the vessel's wall 20 um from where the mask has it.
BOX = CropBox((0, 40, 0), (48, 140, 200), SHAPE)


def _endothelial_image_and_lumen():
    """One wide vessel (lumen radius 6 um) along x at y = 30 um, drawn as a
    wall ring under the E14.5 stack's blur, and its lumen mask."""
    z, y = np.meshgrid(np.arange(SHAPE[0]) * VS[0], np.arange(SHAPE[1]) * VS[1], indexing="ij")
    distance = np.repeat(np.hypot(z - 24, y - 30)[..., None], SHAPE[2], axis=2)
    wall = (distance > 6.0) & (distance <= 7.2)
    image = ndimage.gaussian_filter(np.where(wall, 100.0, 0.0), (2.9, 2.2, 2.2)) + 10.0
    image += np.random.default_rng(0).normal(0.0, 3.0, SHAPE)
    return image.astype(np.float32), distance <= 6.0


def _settings(**overrides):
    return {
        "raw_image_stain": "auto",
        "raw_image_path": "raw.tif",
        "raw_image_channel": None,
        "use_memmap_loading": False,
        "memmap_directory": None,
        "image_axis_order": "zyx",
        **overrides,
    }


def _network(G, mask):
    return SimpleNamespace(graph=G, volume=SimpleNamespace(image=mask, voxel_size_zyx=VS))


def test_a_named_stain_is_taken_as_named_with_nothing_loaded(monkeypatch):
    import haemolynx.haemodynamics.fwhm_diameter as fwhm_diameter

    monkeypatch.setattr(fwhm_diameter, "load_single_channel_tiff_volume", _never)
    network = _network(nx.MultiGraph(), np.zeros((2, 2, 2), dtype=bool))
    assert stages.resolve_raw_image_stain(_settings(raw_image_stain=ENDOTHELIAL), network) == (
        ENDOTHELIAL, None, None
    )


def test_the_builds_answer_is_taken_without_opening_the_raw_image(monkeypatch):
    """build_network recorded what it read the stain as; the Diameters stage
    takes that, so one graph never names its raw image two ways."""
    import haemolynx.haemodynamics.fwhm_diameter as fwhm_diameter

    monkeypatch.setattr(fwhm_diameter, "load_single_channel_tiff_volume", _never)
    G = nx.MultiGraph()
    G.graph["raw_image_stain"] = ENDOTHELIAL
    network = _network(G, np.zeros((2, 2, 2), dtype=bool))

    assert stages.resolve_raw_image_stain(_settings(), network) == (ENDOTHELIAL, None, None)


def test_a_run_that_must_read_cuts_the_raw_image_to_its_crop_box(monkeypatch):
    """Read uncut against the cropped mask, this endothelial image came back
    'cannot tell' or plasma, and was then measured by FWHM."""
    import haemolynx.haemodynamics.fwhm_diameter as fwhm_diameter

    image, lumen = _endothelial_image_and_lumen()
    monkeypatch.setattr(fwhm_diameter, "load_single_channel_tiff_volume", lambda *_a, **_k: image)
    monkeypatch.setattr(stages.io, "resolve_image_path_with_optional_zip", lambda path: path)
    G = nx.MultiGraph()
    G.graph["image_voxel_size_zyx"] = VS
    G.graph[CROP_ATTRIBUTE] = BOX.to_record(VS)
    mask = lumen[BOX.slices]
    network = _network(G, mask)

    stain, reading, raw = stages.resolve_raw_image_stain(_settings(), network)

    assert stain == ENDOTHELIAL and reading.stain == ENDOTHELIAL
    assert raw.shape == mask.shape
    assert np.array_equal(raw, image[BOX.slices])


def test_an_uncropped_run_reads_the_whole_image_and_hands_it_on(monkeypatch):
    import haemolynx.haemodynamics.fwhm_diameter as fwhm_diameter

    image, lumen = _endothelial_image_and_lumen()
    monkeypatch.setattr(fwhm_diameter, "load_single_channel_tiff_volume", lambda *_a, **_k: image)
    monkeypatch.setattr(stages.io, "resolve_image_path_with_optional_zip", lambda path: path)
    G = nx.MultiGraph()
    G.graph["image_voxel_size_zyx"] = VS

    stain, reading, raw = stages.resolve_raw_image_stain(_settings(), _network(G, lumen))

    assert stain == ENDOTHELIAL and raw is image


def _never(*_args, **_kwargs):
    raise AssertionError("the raw image was opened again")


def _cropped_vessel_graph() -> nx.MultiGraph:
    """One vessel in a network built in the box (z, y, x) 5:15, 10:30, 16:48 of
    a 20 x 40 x 64 image of 1 um voxels (tests/test_crop_on_import.py's)."""
    G = nx.MultiGraph()
    G.graph["image_voxel_size_zyx"] = (1.0, 1.0, 1.0)
    G.graph[CROP_ATTRIBUTE] = CropBox((5, 10, 16), (15, 30, 48), (20, 40, 64)).to_record((1.0, 1.0, 1.0))
    G.add_node(0, pos=np.array([5.0, 10.0, 2.0]))
    G.add_node(1, pos=np.array([5.0, 10.0, 30.0]))
    G.add_edge(0, 1, key=0, length=28.0, branch_order="B01",
               voxels=[[5.0, 10.0, float(x)] for x in range(2, 31)])
    return G


def test_fwhm_reads_the_raw_volume_it_is_handed_instead_of_the_file(monkeypatch):
    from haemolynx.haemodynamics import apply
    from haemolynx.haemodynamics.apply import HaemodynamicsApplyConfig, assign_edge_diameters

    cut = np.arange(10 * 20 * 32, dtype=np.float32).reshape(10, 20, 32)
    seen = []

    def fake_fwhm(G, _config, raw_volume=None, **_kwargs):
        seen.append(raw_volume)
        return {}

    monkeypatch.setattr(apply, "_measure_fwhm_diameters", fake_fwhm)
    monkeypatch.setattr(apply, "load_fwhm_raw_volume", _never)
    config = HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": {"B01": 5.0}},
        fwhm={"use_fwhm_edge_diameters": True, "do_fwhm_measurement": True,
              "fwhm_fix_blur_to_image_psf": False},
    )

    _G, _summary, raw = assign_edge_diameters(_cropped_vessel_graph(), config, raw_volume=cut)

    assert seen[0] is cut and raw is cut


def test_the_endothelial_fit_reads_the_raw_volume_it_is_handed_instead_of_the_file(monkeypatch):
    from haemolynx.haemodynamics import apply
    from haemolynx.haemodynamics.apply import HaemodynamicsApplyConfig, assign_edge_diameters

    cut = np.arange(10 * 20 * 32, dtype=np.float32).reshape(10, 20, 32)
    seen = []

    def fake_fit(G, *, endothelial_volume, **_kwargs):
        seen.append(endothelial_volume)
        return {"measured": 0}

    monkeypatch.setattr(apply.endothelial, "measure_edge_diameters_from_endothelium", fake_fit)
    monkeypatch.setattr(apply.fwhm_diameter, "load_single_channel_tiff_volume", _never)
    config = HaemodynamicsApplyConfig(
        diameters={"diameter_by_branch_order": {"B01": 5.0}},
        fwhm={"use_fwhm_edge_diameters": False},
        endothelial={"use_endothelial_diameters": True, "endothelial_image_path": "raw.tif"},
    )

    _G, summary, _raw = assign_edge_diameters(_cropped_vessel_graph(), config, raw_volume=cut)

    assert seen[0] is cut
    assert summary["endothelial"] == {"measured": 0}


def test_the_stain_reader_refuses_a_raw_image_on_another_grid():
    """The guard behind the crop: a raw image not the mask's shape is an
    error, never a reading from the wrong place."""
    from haemolynx.preprocessing.raw_stain import read_raw_stain

    image, lumen = _endothelial_image_and_lumen()
    with pytest.raises(ValueError, match="differ in shape"):
        read_raw_stain(image, lumen[BOX.slices], VS)
    assert read_raw_stain(image[BOX.slices], lumen[BOX.slices], VS).stain == ENDOTHELIAL
    assert read_raw_stain(image, lumen, VS).stain == ENDOTHELIAL
    assert PLASMA_COLUMN != ENDOTHELIAL
