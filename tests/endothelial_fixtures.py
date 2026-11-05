"""Synthetic endothelial stacks: hollow vessels -- a dark lumen inside a thin
bright wall, as a claudin-5 or CD31 channel shows them -- at any voxel size,
wall thickness and blur, with the vessel graph a run would build on them.

Each vessel is a straight tube through the whole volume, drawn on sub-voxel
samples and blurred by the optics, with Poisson noise; its centreline is cut
into connected edges, so their inner ends are junctions as a run's are. The
segmentation is each tube filled out to its outer wall. Drawn independently
of :mod:`haemolynx.haemodynamics.endothelial_planted`, which the tests check
against these.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter

#: Three vessels crossing near the middle: along x, along y, and mostly
#: along z -- a ring tilted towards z for the blur along z to be read from.
#: Each is (direction, centre as a share of the extent, half length in um).
CROSSING_VESSELS = (
    ((0.0, 0.0, 1.0), (0.5, 0.27, 0.5), 24.0),
    ((0.0, 1.0, 0.0), (0.5, 0.5, 0.27), 24.0),
    ((1.0, 0.3, 0.2), (0.5, 0.32, 0.32), 20.0),
)


@dataclass(frozen=True)
class Scene:
    volume: np.ndarray
    mask: np.ndarray
    graph: nx.MultiGraph
    voxel_size_zyx: tuple[float, float, float]
    #: The blur a ring fit sees: the optics, and the voxels' own box.
    psf_seen_zyx: tuple[float, float, float]
    wall_um: float
    lumen_um: float


def psf_seen(optics_um, voxel_size_zyx) -> tuple[float, float, float]:
    return tuple(float(np.hypot(s, v / np.sqrt(12.0))) for s, v in zip(optics_um, voxel_size_zyx))


def hollow_vessel_scene(
    *,
    voxel_size_zyx=(1.0, 0.98, 0.98),
    optics_um=(2.0, 0.5, 0.5),
    wall_um: float = 1.0,
    lumen_um: float = 9.0,
    extent_um=(48.0, 110.0, 110.0),
    vessels=CROSSING_VESSELS,
    pieces: int = 6,
    peak: float = 300.0,
    background: float = 5.0,
    seed: int = 0,
) -> Scene:
    rng = np.random.default_rng(seed)
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    extent = np.asarray(extent_um, dtype=float)
    shape = tuple(int(v) for v in np.ceil(extent / spacing))
    points = np.stack(np.indices(shape), axis=-1).reshape(-1, 3) * spacing
    occupancy = np.zeros(len(points))
    mask = np.zeros(len(points), dtype=bool)
    sub = (np.arange(2) + 0.5) / 2 - 0.5
    offsets = np.array([[dz, dy, dx] for dz in sub for dy in sub for dx in sub]) * spacing
    inner, outer = lumen_um / 2, lumen_um / 2 + wall_um
    lines = []
    for direction, centre_share, half_length in vessels:
        d = np.asarray(direction, dtype=float)
        d /= np.linalg.norm(d)
        centre = np.asarray(centre_share) * extent
        for offset in offsets:
            rel = points + offset - centre
            r = np.linalg.norm(rel - (rel @ d)[:, None] * d, axis=1)
            occupancy += ((r >= inner) & (r <= outer)) / len(offsets)
        rel = points - centre
        mask |= np.linalg.norm(rel - (rel @ d)[:, None] * d, axis=1) <= outer
        lines.append(centre + np.linspace(-half_length, half_length, 25)[:, None] * d)
    blurred = gaussian_filter(occupancy.reshape(shape), np.asarray(optics_um) / spacing)
    volume = rng.poisson(background + peak * blurred).astype(np.float32)

    G = nx.MultiGraph()
    G.graph["image_voxel_size_zyx"] = tuple(float(v) for v in voxel_size_zyx)
    base = 0
    for line in lines:
        cuts = np.linspace(0, len(line) - 1, pieces + 1).round().astype(int)
        for k in range(pieces):
            part = line[cuts[k]:cuts[k + 1] + 1]
            G.add_node(base + k, pos=part[0])
            G.add_node(base + k + 1, pos=part[-1])
            G.add_edge(
                base + k, base + k + 1, voxels=[tuple(p) for p in part],
                length=float(np.sum(np.linalg.norm(np.diff(part, axis=0), axis=1))),
            )
        base += pieces + 1
    return Scene(
        volume=volume,
        mask=mask.reshape(shape),
        graph=G,
        voxel_size_zyx=tuple(float(v) for v in voxel_size_zyx),
        psf_seen_zyx=psf_seen(optics_um, voxel_size_zyx),
        wall_um=float(wall_um),
        lumen_um=float(lumen_um),
    )
