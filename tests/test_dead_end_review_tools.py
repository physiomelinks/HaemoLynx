"""The G4 dead-end grid's candidate measures (``scripts/dead_end_review/build_g4.py``):
how bright a stain reads along a chain against its own background, on volumes
whose answer is known."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dead_end_review import build_g4
from lumen_artefact_fixtures import capsule

VS = (1.0, 1.0, 1.0)
SHAPE = (24, 40, 60)


def _volume_with_a_vessel(brightness=100.0, *, noise=0.0, seed=0):
    """A segmented vessel of radius 2.5 along x at (z, y) = (12, 12), filled
    at *brightness*; the background 10, plus *noise*."""
    rng = np.random.default_rng(seed)
    mask = capsule(np.zeros(SHAPE, dtype=bool), (12, 12, -5), (12, 12, 65), 2.5)
    volume = np.full(SHAPE, 10.0) + rng.normal(0.0, noise, SHAPE) if noise else np.full(SHAPE, 10.0)
    volume[mask] = brightness
    return volume, mask


def _path(y, x0=10.0, x1=40.0):
    return np.column_stack([np.full(61, 12.0), np.full(61, float(y)), np.linspace(x0, x1, 61)])


def test_a_chain_along_a_bright_vessel_reads_its_brightness_over_its_background():
    volume, mask = _volume_with_a_vessel()
    level = build_g4.channel_level(volume, mask, _path(12), 2.5, VS)
    assert level["inside"] == pytest.approx(100.0)
    assert level["background"] == pytest.approx(10.0)
    assert build_g4.channel_contrast(level, reference=90.0) == pytest.approx(1.0)


def test_a_chain_through_background_reads_no_contrast():
    """The case G4 asks about: a chain where no stain says there is a vessel."""
    volume, mask = _volume_with_a_vessel(noise=2.0)
    level = build_g4.channel_level(volume, mask, _path(30), 2.0, VS)
    assert abs(build_g4.channel_contrast(level, reference=90.0)) < 0.1


def test_a_segmented_neighbour_is_left_out_of_the_background():
    """The shell 4-8 um round a chain beside another vessel would read that
    vessel; the mask (grown by 1.5 um) keeps it out of the background."""
    volume, mask = _volume_with_a_vessel()
    level = build_g4.channel_level(volume, mask, _path(19), 1.5, VS)  # 7 um from the vessel's axis
    assert level["background"] == pytest.approx(10.0)


def test_an_empty_path_or_reference_reads_nothing():
    volume, mask = _volume_with_a_vessel()
    assert np.isnan(build_g4.channel_level(volume, mask, np.empty((0, 3)), 2.0, VS)["inside"])
    assert np.isnan(build_g4.channel_contrast({"inside": 5.0, "background": 1.0}, reference=0.0))


def test_the_grid_is_spread_across_both_stains():
    """Ten from each third of the endothelial contrast, each ten spread
    across the dextran contrast."""
    rng = np.random.default_rng(1)
    ends = [SimpleNamespace(tip=i) for i in range(300)]
    measures = {i: {"dextran_tip_contrast": float(rng.uniform(0, 1)), "endothelium_tip_contrast": float(rng.uniform(0, 1))}
                for i in range(300)}
    chosen = build_g4.choose(ends, measures, count=30)
    assert len({e.tip for e in chosen}) == 30
    endo = np.asarray([measures[e.tip]["endothelium_tip_contrast"] for e in chosen])
    cuts = np.percentile([m["endothelium_tip_contrast"] for m in measures.values()], [100 / 3, 200 / 3])
    assert np.histogram(endo, bins=[-np.inf, *cuts, np.inf])[0].tolist() == [10, 10, 10]
    dextran = sorted(measures[e.tip]["dextran_tip_contrast"] for e in chosen)
    assert dextran[0] < 0.1 and dextran[-1] > 0.9
