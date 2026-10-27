"""Tubes of known width, for the size convention (``classify_vessels_by_size``).

One straight vessel along x, square in cross-section, 15, 9 and then 5 voxels
across (odd widths, so a centreline skeletonises down their middle): at 1 um
voxels the mask's cross-section reads a circle of the same area, about 16.9,
10.2 and 5.6 um, so a 6 / 15 um convention classes the three pieces wide, mid
and narrow. :func:`tube_chain` is that vessel as a graph, one edge per piece;
:func:`tube_chain_with_branches` adds a narrow side branch at each change of
width, so a skeleton of it has junctions there and a built graph splits the
trunk where its width changes.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

#: (first x, last x + 1, width in voxels) of each piece of the trunk.
PIECES = ((0, 30, 15), (30, 60, 9), (60, 90, 5))
#: Where the trunk runs, (z, y), and the volume's (z, y) extent.
CENTRE = 10
EXTENT = 21
#: The trunk's nodes along x: an end, the two changes of width, an end.
NODE_X = (2, 30, 60, 88)
#: The class each piece's width is in at 6 / 15 um.
CLASSES = ("wide", "mid", "narrow")


def _square(width: int) -> slice:
    half = width // 2
    return slice(CENTRE - half, CENTRE + half + 1)


def tube_chain_mask(*, branches: bool = False, side_length: int = 30) -> np.ndarray:
    """The trunk as a uint8 mask, (z, y, x); with *branches*, a 5-voxel side
    branch running +y from each change of width to the image's face."""
    shape = (EXTENT, EXTENT + (side_length if branches else 0), PIECES[-1][1])
    mask = np.zeros(shape, dtype=np.uint8)
    for start, stop, width in PIECES:
        across = _square(width)
        mask[across, across, start:stop] = 1
    if branches:
        side = _square(5)
        for x in NODE_X[1:3]:
            mask[side, CENTRE:, x - 2 : x + 3] = 1
    return mask


def tube_chain() -> nx.MultiGraph:
    """The trunk as a graph: nodes 0..3 at :data:`NODE_X`, one edge per piece,
    positions and voxels in microns at 1 um voxels."""
    G = nx.MultiGraph()
    for node, x in enumerate(NODE_X):
        G.add_node(node, pos=np.array([float(CENTRE), float(CENTRE), float(x)]))
    for node in range(len(NODE_X) - 1):
        start, end = NODE_X[node], NODE_X[node + 1]
        voxels = [(float(CENTRE), float(CENTRE), float(x)) for x in range(start, end + 1)]
        G.add_edge(node, node + 1, voxels=voxels, length=float(end - start))
    return G
