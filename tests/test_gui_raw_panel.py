"""``gui/raw_panel.py``: what the Post processing page's raw panel draws.

A rod of segmentation with a bright raw lumen, and items lying in it: the
planes they are drawn in, the pictures read through a slab of them (only a
window of the volume), and where the item, the other centrelines and the
scale bar land on the pictures."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from haemolynx.gui import raw_panel
from haemolynx.gui.raw_panel import (
    ACROSS_IT,
    HALF_THICKNESS_UM,
    MAX_RADIUS_UM,
    MIN_RADIUS_UM,
    OWN_PLANE,
    SCALE_BAR_UM,
    centreline_points,
    edge_lines,
    item_planes,
    panel_views,
    slab_image,
    stretch,
)

VOXEL = (1.0, 1.0, 1.0)
SHAPE = (20, 40, 60)


def _rod():
    """A 5 x 5 voxel rod along x at z = 10, y = 20, and its raw image: bright
    in the rod, dim speckle round it."""
    mask = np.zeros(SHAPE, dtype=bool)
    mask[8:13, 18:23, 10:50] = True
    raw = np.random.default_rng(0).uniform(0.0, 20.0, size=SHAPE).astype(np.float32)
    raw[mask] = 500.0
    return mask, raw


def _flat_item():
    """Points spread in x and y at one depth, z = 10."""
    x = np.linspace(15.0, 45.0, 31)
    return np.column_stack([np.full_like(x, 10.0), 20.0 + 0.2 * (x - 30.0), x])


def test_an_item_lying_in_one_image_plane_is_drawn_as_napari_draws_that_plane():
    own, across = item_planes(_flat_item())
    assert (own.label, across.label) == (OWN_PLANE, ACROSS_IT)
    assert np.allclose(np.abs(own.normal), [1.0, 0.0, 0.0], atol=1e-6)
    centre = own.centre_um
    # x to the right, y down.
    (right,), (down,) = own.fractions(centre + [0, 0, 5]), own.fractions(centre + [0, 5, 0])
    # The item slants 0.2 um in y per um of x, so its longest direction does too.
    assert right[0] > 0.5 and right[1] == pytest.approx(0.5, abs=0.05)
    assert down[1] > 0.5
    # Across it, deeper is lower.
    (deeper,) = across.fractions(centre + [5, 0, 0])
    assert deeper[1] > 0.5 and deeper[0] == pytest.approx(0.5)
    for plane in (own, across):
        assert np.dot(plane.across, plane.up) == pytest.approx(0.0, abs=1e-9)
        assert np.allclose(np.cross(plane.across, plane.up), plane.normal)


def test_an_item_of_one_point_or_one_line_falls_back_on_the_image_axes():
    own, _across = item_planes(np.empty((0, 3)), centre_um=(5.0, 6.0, 7.0))
    assert np.allclose(own.centre_um, [5.0, 6.0, 7.0])
    assert np.allclose(np.abs(own.normal), [1.0, 0.0, 0.0])
    assert own.radius_um == MIN_RADIUS_UM
    line = np.column_stack([np.zeros(5), np.zeros(5), np.linspace(0, 200, 5)])
    own, _across = item_planes(line)
    assert np.allclose(own.across, [0.0, 0.0, 1.0]) and np.allclose(np.abs(own.normal), [1.0, 0.0, 0.0])
    assert own.radius_um == MAX_RADIUS_UM


def test_a_slab_reads_the_mask_either_side_of_the_plane_and_no_further():
    mask, _raw = _rod()
    own, _ = item_planes(_flat_item())
    picture = slab_image(mask, own, VOXEL, reduce="any")
    assert picture.shape[0] == picture.shape[1]
    middle = picture.shape[0] // 2
    assert picture[middle, middle] == 1.0, "the rod runs through the middle"
    assert picture[2, 2] == 0.0
    # A plane 5 um above the rod's top layer misses it at 3 um either side;
    # 2 um above it does not.
    for lift, seen in ((2.0 + 2.0, True), (2.0 + 5.0, False)):
        plane = raw_panel.Plane(OWN_PLANE, own.centre_um + lift * own.normal, own.across, own.up, own.normal,
                                own.radius_um)
        assert bool(slab_image(mask, plane, VOXEL, reduce="any").any()) is seen, lift


def test_a_slab_reads_only_a_window_of_the_volume():
    """A memory-mapped raw image stays on disk: one read, of a window round
    the plane, giving what the whole array gives."""
    mask, raw = _rod()
    big = np.zeros((20, 200, 400), dtype=np.float32)
    big[:, :40, :60] = raw
    reads = []

    class Watched:
        shape = big.shape

        def __getitem__(self, key):
            reads.append(key)
            return big[key]

    own, _ = item_planes(_flat_item())
    picture = slab_image(Watched(), own, VOXEL)
    assert picture == pytest.approx(slab_image(big, own, VOXEL))
    (key,) = reads
    window = np.prod([s.stop - s.start for s in key])
    assert window < 0.05 * big.size
    assert picture.max() > 100.0 and picture.min() < 20.0


def test_stretch_sets_the_contrast_and_a_flat_picture_is_black():
    assert not stretch(np.full((4, 4), 7.0)).any()
    ramp = stretch(np.arange(100.0).reshape(10, 10))
    assert ramp.min() == 0.0 and ramp.max() == 1.0


def _graph():
    G = nx.MultiGraph()
    for node, pos in {0: (10, 20, 10), 1: (10, 20, 49), 2: (10, 35, 30), 3: (2, 5, 5)}.items():
        G.add_node(node, pos=np.asarray(pos, dtype=float))
    G.add_edge(0, 1, voxels=[(10.0, 20.0, x) for x in range(10, 50)])
    G.add_edge(2, 0, voxels=[(10.0, 35.0 - t, 30.0 - t) for t in range(0, 16)] + [(10.0, 20.0, 10.0)])
    return G


def test_the_item_is_drawn_from_its_vessels_and_the_rest_as_dots_near_the_slab():
    mask, raw = _rod()
    G = _graph()
    lines = edge_lines(G, [(0, 1, 0), (5, 6, 0)])  # one edge no longer there
    assert len(lines) == 1
    others = centreline_points(G)
    views = panel_views(lines[0], voxel_size_zyx=VOXEL, mask=mask, raw=raw, item_lines_um=lines,
                        other_points_um=others)
    own = views[0]
    assert own.plane.label == OWN_PLANE and own.segmentation is not None and own.raw is not None
    assert 0.0 <= own.raw.min() and own.raw.max() <= 1.0
    (line,) = own.item_lines
    assert np.all((line >= 0) & (line <= 1)) and np.ptp(line[:, 0]) > 0.5
    assert len(own.item_dots) == 0, "an item with vessels is drawn as lines"
    assert len(own.other_dots) > 0
    x0, x1, _y = own.scale_bar
    assert x1 - x0 == pytest.approx(SCALE_BAR_UM / (2 * own.plane.radius_um))
    # Without a raw image, its picture is left out.
    assert panel_views(lines[0], voxel_size_zyx=VOXEL, mask=mask)[0].raw is None


def test_an_item_without_vessels_is_drawn_as_its_points_in_the_slab():
    blob = np.array([[10.0, 20.0, 30.0], [10.0, 21.0, 31.0], [10.0 + HALF_THICKNESS_UM + 1, 20.0, 30.0]])
    (own, _across) = panel_views(blob, voxel_size_zyx=VOXEL)
    assert own.segmentation is None and own.raw is None
    assert len(own.item_dots) >= 2
