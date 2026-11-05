"""skan's side: the mask Lee-thinned and summarised by skan, nothing cleaned.

skan (Nunez-Iglesias et al. 2018, BSD-3) is the library HaemoLynx's own graph
building starts from (``graph.build`` reads skan's ``Skeleton``), so on the
raw skeleton the two share skan's pixel graph. What they do not share is
everything after it -- the clean-up, the smoothing, and every statistic -- and
skan's ``summarize`` measures branch length, chord and mean radius itself.
"""
from __future__ import annotations

import inspect

import numpy as np
from scipy import ndimage as ndi

from .measures import Measurements, tortuosity_of

DEFINITIONS = {
    "skeleton": "Lee thinning of the mask as given (skimage `skeletonize(method='lee')`), no clean-up",
    "segment": "a skan path: skeleton pixels between two pixels whose neighbour count is not 2",
    "length": "skan `branch-distance`: sum of the steps between neighbouring pixels, in microns",
    "tortuosity": "`branch-distance` over skan `euclidean-distance` between the path's two ends",
    "diameter": "twice skan `mean-pixel-value` over the path of the mask's distance transform (to the nearest background voxel's centre)",
    "branch point": "a node of skan's branch graph with degree >= 3 -- every junction pixel, so a cluster of touching junction pixels counts each one",
    "end point": "a node of skan's branch graph with degree 1",
}


def junction_clusters(skeleton_graph, shape) -> int:
    """Touching junction pixels counted as one: the branch points a person
    would mark."""
    skeleton = np.zeros(shape, dtype=bool)
    junction_pixels = np.flatnonzero(skeleton_graph.degrees >= 3)
    coordinates = np.round(skeleton_graph.coordinates[junction_pixels]).astype(int)
    skeleton[tuple(coordinates.T)] = True
    return int(ndi.label(skeleton, structure=np.ones((3,) * skeleton.ndim))[1])


def lee_skeleton(mask: np.ndarray) -> np.ndarray:
    """The mask's Lee skeleton as skimage draws it, nothing cleaned."""
    from skimage.morphology import skeletonize

    return skeletonize(np.asarray(mask) > 0, method="lee") > 0


def run(mask: np.ndarray, voxel_size_zyx, *, skeleton: np.ndarray | None = None) -> Measurements:
    """skan's measurements of *mask*'s Lee skeleton, or of *skeleton* when
    given (HaemoLynx's cleaned one, for :func:`haemolynx_side.ledger`)."""
    from skan import Skeleton, summarize

    mask = np.asarray(mask) > 0
    spacing = tuple(float(v) for v in voxel_size_zyx)
    if skeleton is None:
        skeleton = lee_skeleton(mask)
    skeleton = np.asarray(skeleton) > 0
    distance = ndi.distance_transform_edt(mask, sampling=spacing)
    # skan reads each pixel's value off the skeleton image itself, so the
    # skeleton carries the distance transform. A skeleton voxel off the mask
    # (a cleaned skeleton's bridge across a dropout) gets a sliver of a value
    # rather than 0, which would drop it out of the skeleton.
    graph = Skeleton(np.where(skeleton, np.maximum(distance, 1e-6), 0.0), spacing=spacing)
    # skan 0.11 names its columns with "-" and takes no separator; later
    # releases take one (and warn without it).
    if "separator" in inspect.signature(summarize).parameters:
        branches = summarize(graph, separator="-")
    else:
        branches = summarize(graph)
    lengths = branches["branch-distance"].to_numpy(float)
    chords = branches["euclidean-distance"].to_numpy(float)
    tortuosity, loops = tortuosity_of(lengths, chords)
    ends = np.concatenate([branches["node-id-src"].to_numpy(), branches["node-id-dst"].to_numpy()])
    _ids, degree = np.unique(ends, return_counts=True)
    return Measurements(
        tool="skan",
        total_length_um=float(lengths.sum()),
        segments=int(len(branches)),
        branch_points=int(np.count_nonzero(degree >= 3)),
        end_points=int(np.count_nonzero(degree == 1)),
        segment_lengths_um=lengths,
        tortuosity=tortuosity,
        diameters_um=2.0 * branches["mean-pixel-value"].to_numpy(float),
        loops=loops,
        definitions=dict(DEFINITIONS),
        extra={"Junction-pixel clusters": junction_clusters(graph, skeleton.shape)},
    )
