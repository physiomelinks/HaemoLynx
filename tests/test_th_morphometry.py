"""H1 sections 1.3 and 1.5: the TH-dependent morphometrics.

Both reduce to two primitives that can be checked against hand arithmetic: the length of a
centreline, and the distance from a tissue voxel to the nearest centreline. The synthetic
cases below are chosen so the right answer is a number rather than a plausible shape.
"""
import numpy as np
import pytest

from ImageLynx.statistics.th_morphometry import (
    centreline_length_um,
    tissue_to_vessel_distance_um,
)

VOX = (1.8639, 1.866, 1.866)


def test_a_straight_run_measures_its_own_length():
    """n voxels in a row span (n - 1) steps, not n."""
    sk = np.zeros((5, 5, 20), bool)
    sk[2, 2, 3:13] = True                       # 10 voxels along x
    assert centreline_length_um(sk, VOX) == pytest.approx(9 * 1.866)


def test_length_uses_the_axis_specific_spacing():
    """The z spacing differs from y and x, so a z-run is not a y-run."""
    sk_z = np.zeros((20, 5, 5), bool); sk_z[3:13, 2, 2] = True
    sk_y = np.zeros((5, 20, 5), bool); sk_y[2, 3:13, 2] = True
    assert centreline_length_um(sk_z, VOX) == pytest.approx(9 * 1.8639)
    assert centreline_length_um(sk_y, VOX) == pytest.approx(9 * 1.866)
    assert centreline_length_um(sk_z, VOX) != centreline_length_um(sk_y, VOX)


def test_a_diagonal_run_is_longer_than_a_straight_one():
    """Counting voxels would call these equal, which is the error being avoided."""
    n = 8
    diag = np.zeros((12, 12, 12), bool)
    for i in range(n):
        diag[2 + i, 2 + i, 2 + i] = True
    straight = np.zeros((12, 12, 12), bool)
    straight[2, 2, 2:2 + n] = True

    step = np.sqrt(1.8639 ** 2 + 1.866 ** 2 + 1.866 ** 2)
    assert centreline_length_um(diag, VOX) == pytest.approx((n - 1) * step)
    assert centreline_length_um(diag, VOX) > 1.7 * centreline_length_um(straight, VOX)


def test_an_isolated_voxel_has_no_length():
    sk = np.zeros((5, 5, 5), bool)
    sk[2, 2, 2] = True
    assert centreline_length_um(sk, VOX) == 0.0
    assert centreline_length_um(np.zeros((5, 5, 5), bool), VOX) == 0.0


def test_length_within_a_mask_counts_only_fully_enclosed_steps():
    """A step straddling the boundary belongs to neither side, so it is not counted."""
    sk = np.zeros((5, 5, 20), bool)
    sk[2, 2, 3:13] = True                       # 9 steps
    inside = np.zeros((5, 5, 20), bool)
    inside[2, 2, 3:8] = True                    # first 5 voxels, 4 fully enclosed steps

    assert centreline_length_um(sk, VOX, within=inside) == pytest.approx(4 * 1.866)
    assert centreline_length_um(sk, VOX, within=np.ones_like(sk)) == \
        pytest.approx(centreline_length_um(sk, VOX))
    assert centreline_length_um(sk, VOX, within=np.zeros_like(sk)) == 0.0


def test_distance_to_the_centreline_is_measured_in_micrometres():
    sk = np.zeros((9, 9, 9), bool)
    sk[4, 4, 4] = True
    tissue = np.zeros((9, 9, 9), bool)
    tissue[4, 4, 7] = True                      # 3 voxels away along x
    tissue[4, 7, 4] = True                      # 3 voxels away along y
    tissue[7, 4, 4] = True                      # 3 voxels away along z

    d = np.sort(tissue_to_vessel_distance_um(tissue, sk, VOX))
    assert d.size == 3
    assert d[0] == pytest.approx(3 * 1.8639, rel=1e-6)
    assert d[1] == pytest.approx(3 * 1.866, rel=1e-6)
    assert d[2] == pytest.approx(3 * 1.866, rel=1e-6)


def test_a_tissue_voxel_on_the_centreline_is_at_zero_distance():
    sk = np.zeros((7, 7, 7), bool)
    sk[3, 3, 2:5] = True
    tissue = sk.copy()
    assert tissue_to_vessel_distance_um(tissue, sk, VOX).max() == pytest.approx(0.0)


def test_distance_is_to_the_nearest_centreline_not_the_first():
    sk = np.zeros((5, 5, 21), bool)
    sk[2, 2, 0] = True
    sk[2, 2, 20] = True
    tissue = np.zeros((5, 5, 21), bool)
    tissue[2, 2, 18] = True                     # 18 from the left, 2 from the right
    assert tissue_to_vessel_distance_um(tissue, sk, VOX)[0] == pytest.approx(2 * 1.866)


def test_no_tissue_and_no_vessel_are_reported_rather_than_crashing():
    empty = np.zeros((5, 5, 5), bool)
    sk = np.zeros((5, 5, 5), bool)
    sk[2, 2, 2] = True
    assert tissue_to_vessel_distance_um(empty, sk, VOX).size == 0
    with pytest.raises(ValueError, match="no centreline"):
        tissue_to_vessel_distance_um(sk, empty, VOX)
