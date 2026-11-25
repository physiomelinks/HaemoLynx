"""Vessel phantoms of known topology (``graph.8a``).

Each phantom is a network written down in microns -- nodes, and vessels as
straight tubes of a known radius between them -- rasterised into a mask on a
chosen voxel grid. The truth is exact: where every junction is, how long the
centrelines are inside the image, and how many loops the network has. Every
vessel's open end lies outside the box, so the image cuts it at a face as a
real stack cuts its vessels, and the network has no blind end inside.

Two shapes: a branching tree (an arteriole dividing twice) and a mesh (two
hexagonal loops sharing a side, as in a capillary bed). Each is rasterised
at two voxel sizes (isotropic 1 um, and 1 x 1 x 3 um as a coarse z) and two
image qualities: ``clean``, the exact mask; ``noisy``, a raw image (the
tubes blurred by a Gaussian PSF, with noise) thresholded at half its
height, so the mask's surface is rough as a segmentation's is. Generated in
code from a seed: no file needs keeping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Phantom:
    name: str
    #: The box, (z, y, x) in microns, from the origin.
    size_um: tuple[float, float, float]
    #: Node positions, (z, y, x) in microns. Ends may lie outside the box.
    nodes: dict[str, tuple[float, float, float]]
    #: Vessels: (node, node, radius in microns).
    vessels: tuple[tuple[str, str, float], ...]

    def degree(self, node: str) -> int:
        return sum(node in (u, v) for u, v, _ in self.vessels)

    def junctions(self) -> list[str]:
        return sorted(node for node in self.nodes if self.degree(node) >= 3)

    def junction_radius(self, node: str) -> float:
        """The widest vessel meeting at *node*: how far its centre can sit
        from where the skeleton puts the junction."""
        return max(radius for u, v, radius in self.vessels if node in (u, v))

    def inside(self, point: np.ndarray) -> bool:
        return bool(np.all(point >= 0.0) and np.all(point <= np.asarray(self.size_um)))

    def length_inside_um(self) -> float:
        """The centrelines' total length inside the box."""
        total = 0.0
        for u, v, _ in self.vessels:
            total += _clipped_length(np.asarray(self.nodes[u]), np.asarray(self.nodes[v]), np.asarray(self.size_um))
        return total

    def loops(self) -> int:
        """The first Betti number of the part inside the box: vessels minus
        nodes plus pieces, with each end outside the box counted as a node."""
        parent = {node: node for node in self.nodes}

        def root(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for u, v, _ in self.vessels:
            parent[root(u)] = root(v)
        pieces = len({root(node) for node in self.nodes})
        return len(self.vessels) - len(self.nodes) + pieces


def _clipped_length(a: np.ndarray, b: np.ndarray, size: np.ndarray) -> float:
    """The length of segment a-b inside the box [0, size] (Liang-Barsky)."""
    direction = b - a
    start, stop = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if a[axis] < 0.0 or a[axis] > size[axis]:
                return 0.0
            continue
        t0 = (0.0 - a[axis]) / direction[axis]
        t1 = (size[axis] - a[axis]) / direction[axis]
        start, stop = max(start, min(t0, t1)), min(stop, max(t0, t1))
    return max(0.0, stop - start) * float(np.linalg.norm(direction))


def tree() -> Phantom:
    """An arteriole entering at x = 0, dividing into two, each dividing again;
    the four daughters leave through the far faces."""
    nodes = {
        "in": (60.0, 60.0, -10.0),
        "j1": (60.0, 60.0, 40.0),
        "j2": (60.0, 30.0, 75.0),
        "j3": (60.0, 92.0, 72.0),
        "a": (20.0, 5.0, 130.0),
        "b": (100.0, 20.0, 130.0),
        "c": (30.0, 130.0, 100.0),
        "d": (95.0, 105.0, 130.0),
    }
    vessels = (
        ("in", "j1", 5.0),
        ("j1", "j2", 4.0),
        ("j1", "j3", 4.0),
        ("j2", "a", 3.0),
        ("j2", "b", 3.0),
        ("j3", "c", 3.0),
        ("j3", "d", 3.0),
    )
    return Phantom("tree", (120.0, 120.0, 120.0), nodes, vessels)


def mesh() -> Phantom:
    """Two hexagonal loops sharing a side, tilted out of one plane, with six
    vessels leaving through the faces: every junction meets three vessels."""
    z0 = 60.0
    nodes = {
        # The shared side, then each loop's other four corners.
        "s1": (z0 - 6.0, 45.0, 60.0), "s2": (z0 + 6.0, 75.0, 60.0),
        "l1": (z0 - 10.0, 30.0, 35.0), "l2": (z0, 45.0, 10.0 + 20.0), "l3": (z0 + 10.0, 75.0, 35.0 - 5.0),
        "r1": (z0 - 10.0, 30.0, 85.0), "r2": (z0, 45.0, 100.0), "r3": (z0 + 10.0, 75.0, 90.0),
        "lo1": (z0 - 10.0, -10.0, 30.0), "lo2": (z0 + 5.0, 60.0, -10.0), "lo3": (z0 + 15.0, 130.0, 25.0),
        "ro1": (z0 - 10.0, -10.0, 95.0), "ro2": (z0, 50.0, 130.0), "ro3": (z0 + 15.0, 130.0, 100.0),
    }
    radius = 3.0
    vessels = (
        ("s1", "s2", radius),
        ("s1", "l1", radius), ("l1", "l2", radius), ("l2", "l3", radius), ("l3", "s2", radius),
        ("s1", "r1", radius), ("r1", "r2", radius), ("r2", "r3", radius), ("r3", "s2", radius),
        ("l1", "lo1", radius), ("l2", "lo2", radius), ("l3", "lo3", radius),
        ("r1", "ro1", radius), ("r2", "ro2", radius), ("r3", "ro3", radius),
    )
    return Phantom("mesh", (120.0, 120.0, 120.0), nodes, vessels)


PHANTOMS = {"tree": tree, "mesh": mesh}

#: Voxel sizes, (z, y, x) in microns: isotropic, and a z three times coarser.
VOXEL_SIZES = {"iso": (1.0, 1.0, 1.0), "z3": (3.0, 1.0, 1.0)}

#: Image qualities: the PSF's sigma (um) and the noise's standard deviation,
#: on a raw image whose vessels are 1 and background 0.
QUALITIES = {"clean": (0.5, 0.02), "noisy": (1.0, 0.15)}


def _distance_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = b - a
    t = np.clip(((points - a) @ direction) / max(float(direction @ direction), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[..., None] * direction), axis=-1)


def tubes(phantom: Phantom, voxel_size_zyx: tuple[float, float, float]) -> np.ndarray:
    """The exact mask: each voxel whose centre lies within a vessel's radius
    of its centreline. Voxel i's centre is at i times the voxel size."""
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    shape = tuple(int(np.floor(size / step)) + 1 for size, step in zip(phantom.size_um, spacing))
    mask = np.zeros(shape, dtype=bool)
    for u, v, radius in phantom.vessels:
        a, b = np.asarray(phantom.nodes[u], dtype=float), np.asarray(phantom.nodes[v], dtype=float)
        low = np.maximum(np.floor((np.minimum(a, b) - radius) / spacing).astype(int), 0)
        high = np.minimum(np.ceil((np.maximum(a, b) + radius) / spacing).astype(int) + 1, shape)
        if np.any(high <= low):
            continue
        grid = np.stack(np.meshgrid(*(np.arange(lo, hi) * step for lo, hi, step in zip(low, high, spacing)), indexing="ij"), axis=-1)
        region = tuple(slice(lo, hi) for lo, hi in zip(low, high))
        mask[region] |= _distance_to_segment(grid, a, b) <= radius
    return mask


def raw_image(mask: np.ndarray, voxel_size_zyx, psf_sigma_um: float, noise: float, seed: int) -> np.ndarray:
    """The mask blurred by an isotropic Gaussian PSF (*psf_sigma_um*, in
    microns, so wider in voxels along a fine axis), plus Gaussian noise."""
    from scipy.ndimage import gaussian_filter

    sigma = [psf_sigma_um / step for step in voxel_size_zyx]
    blurred = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    rng = np.random.default_rng(seed)
    return blurred + rng.normal(0.0, noise, size=mask.shape).astype(np.float32)


def render(
    phantom: Phantom, voxel_size_zyx, quality: str, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """``(mask, raw)``: the clean quality's mask is exact, the noisy one's
    is its raw image thresholded at half the vessels' brightness."""
    sigma, noise = QUALITIES[quality]
    exact = tubes(phantom, voxel_size_zyx)
    raw = raw_image(exact, voxel_size_zyx, sigma, noise, seed)
    mask = exact if quality == "clean" else raw > 0.5
    return mask.astype(np.uint8), raw


def truth(phantom: Phantom) -> dict:
    """What a graph built from the phantom should have, in microns."""
    return {
        "name": phantom.name,
        "junctions": [
            {"node": node, "pos_zyx_um": list(phantom.nodes[node]), "radius_um": phantom.junction_radius(node)}
            for node in phantom.junctions()
        ],
        "length_um": phantom.length_inside_um(),
        "loops": phantom.loops(),
    }


def variants(names: Optional[list[str]] = None) -> list[tuple[str, str, str]]:
    """Every (phantom, voxel size, quality) combination ``graph.8a`` asks for."""
    return [
        (name, voxels, quality)
        for name in (names or list(PHANTOMS))
        for voxels in VOXEL_SIZES
        for quality in QUALITIES
    ]
