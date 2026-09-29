"""H1 sections 1.3 and 1.5: the TH-dependent morphometrics.

Section 1.3's length is the batch network's own edge polylines (open item 40), and section
1.5's distance is to the pipeline skeleton. The synthetic cases below are chosen so the right
answer is a number rather than a plausible shape.
"""
import networkx as nx
import numpy as np
import pytest

from ImageLynx.statistics.th_morphometry import (
    summarise,
    tissue_to_vessel_distance_um,
)

VOX = (1.8639, 1.866, 1.866)


def _graph(polylines):
    """A MultiGraph whose edges carry µm polylines, as the batch network_graph.pkl does."""
    G = nx.MultiGraph()
    for name, pts in polylines.items():
        pts = np.asarray(pts, dtype=float)
        a, b = f"{name}_a", f"{name}_b"
        G.add_node(a, pos=pts[0])
        G.add_node(b, pos=pts[-1])
        G.add_edge(a, b, voxels=[tuple(p) for p in pts])
    return G


def _summary(G, th, shape=(20, 20, 20)):
    skeleton = np.zeros(shape, bool)
    skeleton[10, 10, 10] = True
    return summarise(
        specimen_id="T", group="WKY", graph=G, th_mask=th,
        vessel_mask=np.zeros(shape, bool), skeleton=skeleton, voxel_um=VOX,
        th_threshold=0.5, vessel_threshold=0.95)


def _centres(*indices):
    """Voxel indices to µm voxel centres, the graph's convention."""
    return [tuple(np.asarray(i, float) * np.asarray(VOX)) for i in indices]


def test_the_length_is_the_sum_of_the_edge_polylines():
    G = _graph({"a": _centres((2, 2, 2), (2, 2, 12)),
                "b": _centres((5, 5, 5), (5, 15, 5))})
    r = _summary(G, np.zeros((20, 20, 20), bool))
    assert r.centreline_length_um == pytest.approx(10 * 1.866 + 10 * 1.866)
    assert r.centreline_length_within_th_um == 0.0


def test_an_l_shaped_corner_is_counted_once():
    """The item 17 regression: a corner is two legs, not the three links a voxel-pair sum sees.

    Summing every 26-adjacent pair of skeleton voxels counted the diagonal between the two
    voxels either side of a corner as well as both legs, 9 to 28% over the path length.
    """
    G = _graph({"L": _centres((4, 4, 4), (4, 4, 10), (4, 10, 10))})
    r = _summary(G, np.zeros((20, 20, 20), bool))
    assert r.centreline_length_um == pytest.approx(12 * 1.866)


def test_the_length_within_th_is_the_part_inside_a_slab():
    th = np.zeros((20, 20, 20), bool)
    th[:, :, 6:11] = True                      # x voxels 6 to 10, i.e. 5.5 to 10.5 voxels
    G = _graph({"a": _centres((5, 5, 1), (5, 5, 16))})
    r = _summary(G, th)
    assert r.centreline_length_um == pytest.approx(15 * 1.866)
    # Sampled every half of the finest voxel, so each of the two crossings is placed to within
    # one sample step (0.93 µm).
    assert r.centreline_length_within_th_um == pytest.approx(5 * 1.866, abs=0.5 * 1.8639 * 2)
    assert r.length_density_mm_per_mm3 == pytest.approx(
        r.centreline_length_within_th_um / r.th_volume_um3 * 1e6)


def test_masks_on_another_grid_are_refused():
    G = _graph({"a": _centres((2, 2, 2), (2, 2, 12))})
    with pytest.raises(ValueError, match="skeleton has shape"):
        summarise(
            specimen_id="T", group="WKY", graph=G, th_mask=np.zeros((20, 20, 20), bool),
            vessel_mask=np.zeros((20, 20, 20), bool), skeleton=np.zeros((10, 10, 10), bool),
            voxel_um=VOX, th_threshold=0.5, vessel_threshold=0.95)


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
