"""Telling a plasma (luminal) label from an endothelial (wall) one
(``preprocessing/raw_stain.py``, ``raw_image_stain="auto"``), on synthetic
vessels drawn with the E14.5 stack's blur."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from haemolynx.preprocessing.raw_stain import ENDOTHELIAL, PLASMA_COLUMN, read_raw_stain

VS = (1.0, 0.5, 0.5)
SHAPE = (48, 160, 160)
#: Sigma (z, y, x) in voxels: about 2.9 um along z and 1.1 um across, the
#: E14.5 claudin-5 channel's own calibration.
BLUR = (2.9, 2.2, 2.2)


def _vessels():
    """Three vessels along x of lumen radius 3, 4 and 6 um, and their radial
    distance from the axis in microns."""
    z, y = np.meshgrid(np.arange(SHAPE[0]) * VS[0], np.arange(SHAPE[1]) * VS[1], indexing="ij")
    distance = np.full(SHAPE[:2], np.inf)
    radius = np.zeros(SHAPE[:2])
    for (cz, cy), r in (((24, 20), 3.0), ((24, 40), 4.0), ((24, 62), 6.0)):
        d = np.hypot(z - cz, y - cy)
        closer = d < distance
        distance[closer], radius[closer] = d[closer], r
    distance = np.repeat(distance[..., None], SHAPE[2], axis=2)
    radius = np.repeat(radius[..., None], SHAPE[2], axis=2)
    return distance, radius


def _image(stain: str, *, noise=3.0, seed=0):
    distance, radius = _vessels()
    if stain == PLASMA_COLUMN:
        image = np.where(distance <= radius, 100.0, 0.0)
    else:  # a 1.2 um wall ring round the lumen
        image = np.where((distance > radius) & (distance <= radius + 1.2), 100.0, 0.0)
    image = ndimage.gaussian_filter(image, BLUR) + 10.0
    return image + np.random.default_rng(seed).normal(0.0, noise, SHAPE)


def _lumen_mask():
    distance, radius = _vessels()
    return distance <= radius


@pytest.mark.parametrize("stain", [PLASMA_COLUMN, ENDOTHELIAL])
def test_each_stain_is_told_apart_from_the_lumen_mask(stain):
    reading = read_raw_stain(_image(stain), _lumen_mask(), VS)
    assert reading.stain == stain
    assert abs(reading.contrast) > 0.3
    assert reading.blocks > 0 and reading.core_voxels > 0 and reading.wall_voxels > 0
    assert ("plasma" in reading.describe()) == (stain == PLASMA_COLUMN)


def test_an_image_unrelated_to_the_mask_says_it_cannot_tell():
    noise = 10.0 + np.random.default_rng(1).normal(0.0, 3.0, SHAPE)
    assert read_raw_stain(noise, _lumen_mask(), VS).stain is None


def test_no_vessel_wide_enough_to_have_a_middle_says_it_cannot_tell():
    """Across a capillary the blur fills a wall ring in: only vessels with a
    core are read, and with none there is nothing to read."""
    thin = _vessels()[0] <= 1.0
    reading = read_raw_stain(_image(PLASMA_COLUMN), thin, VS)
    assert reading.stain is None and reading.blocks == 0


def test_an_empty_mask_reads_nothing():
    reading = read_raw_stain(_image(PLASMA_COLUMN), np.zeros(SHAPE, dtype=bool), VS)
    assert reading.stain is None and reading.blocks == 0
