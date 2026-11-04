"""The filter bank a pixel classifier reads, computed a box of voxels at a time.

Each channel of the raw image is first scaled to its own intensity range
(:func:`intensity_range`: two percentiles of a fixed sample of its voxels), so
a classifier trained on one image reads another image of the same stain at
another brightness alike. Then, per channel: the scaled intensity, and at each
scale (a Gaussian's sigma in microns, so z's coarser spacing is read in the
same units as y and x) the smoothed intensity, its gradient magnitude, its
Laplacian and the three eigenvalues of its Hessian, largest first: what
ilastik's default features measure, on one Gaussian per scale with its
derivatives taken by central differences.

A voxel's features depend only on the voxels within :func:`feature_halo` of
it, so a box computed with that much margin gives, on its core, exactly the
values the whole volume gives -- bit for bit, at the volume's own faces too.
Training and segmenting therefore read the same numbers however the volume is
cut into boxes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from scipy import ndimage as ndi

#: The scales (Gaussian sigma, microns) the training window offers.
SCALE_CHOICES_UM = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

#: The scales a new classifier starts with: capillaries to small arterioles.
DEFAULT_SCALES_UM = (1.0, 2.0, 4.0, 8.0)

#: How many sigmas each Gaussian reaches.
TRUNCATE = 3.0

#: What each scale contributes, per channel, in this order.
PER_SCALE = ("smoothed", "gradient", "laplacian", "hessian_1", "hessian_2", "hessian_3")

#: The intensities a channel's range is read at (percent).
RANGE_PERCENTILES = (1.0, 99.9)

#: About how many voxels the range is read from: an even lattice through the
#: whole volume, the same lattice for the same shape.
RANGE_SAMPLE_VOXELS = 4_000_000

#: Voxels per padded box when a caller does not say.
BOX_VOXELS = 4_000_000


@dataclass(frozen=True)
class FeatureSpec:
    """Which features a classifier reads.

    *channels* are the raw image's channels, counting from 0 as
    ``raw_image_channel`` does; ``(None,)`` is a file with one channel.
    """

    scales_um: tuple[float, ...] = DEFAULT_SCALES_UM
    channels: tuple[int | None, ...] = (None,)

    def __post_init__(self):
        scales = tuple(float(scale) for scale in self.scales_um)
        if not scales or any(not math.isfinite(s) or s <= 0 for s in scales):
            raise ValueError(f"Feature scales must be positive microns, got {self.scales_um!r}.")
        if len(set(scales)) != len(scales):
            raise ValueError(f"Feature scales are listed twice: {self.scales_um!r}.")
        channels = tuple(None if c is None else int(c) for c in self.channels)
        if not channels:
            raise ValueError("A classifier reads at least one channel.")
        if None in channels and len(channels) > 1:
            raise ValueError("The whole file (no channel) cannot be read beside a channel.")
        if len(set(channels)) != len(channels):
            raise ValueError(f"Channels are listed twice: {self.channels!r}.")
        object.__setattr__(self, "scales_um", tuple(sorted(scales)))
        object.__setattr__(self, "channels", channels)

    @property
    def n_features(self) -> int:
        return len(self.channels) * (1 + len(PER_SCALE) * len(self.scales_um))

    def names(self) -> list[str]:
        """Each feature's name, in the order a feature array holds them."""
        names = []
        for channel in self.channels:
            prefix = "image" if channel is None else f"C{channel + 1}"
            names.append(f"{prefix} intensity")
            for scale in self.scales_um:
                names += [f"{prefix} {kind} {scale:g} um" for kind in PER_SCALE]
        return names

    def to_record(self) -> dict:
        return {"scales_um": list(self.scales_um), "channels": list(self.channels)}

    @classmethod
    def from_record(cls, record: dict) -> "FeatureSpec":
        return cls(
            scales_um=tuple(record["scales_um"]),
            channels=tuple(record.get("channels", [None])),
        )


def sigma_voxels(scale_um: float, voxel_size_zyx: Sequence[float]) -> tuple[float, float, float]:
    """A scale in microns as a Gaussian sigma in voxels along z, y and x."""
    return tuple(float(scale_um) / float(size) for size in voxel_size_zyx)  # type: ignore[return-value]


def feature_halo(spec: FeatureSpec, voxel_size_zyx: Sequence[float]) -> tuple[int, int, int]:
    """The margin (voxels, per axis) a box needs for its core's features to
    be the whole volume's: the widest Gaussian's reach (as ``scipy.ndimage``
    rounds it), and two more for the second differences."""
    widest = max(spec.scales_um)
    return tuple(  # type: ignore[return-value]
        int(TRUNCATE * sigma + 0.5) + 2 for sigma in sigma_voxels(widest, voxel_size_zyx)
    )


def intensity_range(volume: np.ndarray) -> tuple[float, float]:
    """``(low, high)``: the :data:`RANGE_PERCENTILES` of an even lattice of
    about :data:`RANGE_SAMPLE_VOXELS` of *volume*'s voxels. The same volume
    always gives the same range, a slab of the lattice read at a time."""
    shape = tuple(int(n) for n in np.shape(volume))
    total = int(np.prod(shape, dtype=np.int64))
    if total == 0:
        raise ValueError("The image is empty.")
    step = max(1, int(math.ceil((total / RANGE_SAMPLE_VOXELS) ** (1.0 / len(shape)))))
    sample = np.concatenate([
        np.asarray(volume[z, ::step, ::step], dtype=np.float32).ravel()
        for z in range(0, shape[0], step)
    ])
    low, high = (float(v) for v in np.percentile(sample, RANGE_PERCENTILES))
    if not high > low:
        high = low + 1.0
    return low, high


def scaled(box: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    """*box* as float32 on its channel's range: *low* 0, *high* 1."""
    low, high = value_range
    out = np.asarray(box, dtype=np.float32) - np.float32(low)
    out /= np.float32(high - low)
    return out


def _difference(values: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    """Central differences along *axis* per micron (one-sided at the ends),
    0 along an axis one voxel long (a 2D image)."""
    if values.shape[axis] < 2:
        return np.zeros_like(values)
    return np.gradient(values, float(spacing), axis=axis)


def symmetric_eigenvalues(a11, a22, a33, a12, a13, a23) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The eigenvalues of symmetric 3x3 matrices, largest first, in closed
    form (Smith's trigonometric solution): per voxel, with no Python loop."""
    q = (a11 + a22 + a33) / 3
    b11, b22, b33 = a11 - q, a22 - q, a33 - q
    off = a12 * a12 + a13 * a13 + a23 * a23
    p = np.sqrt((b11 * b11 + b22 * b22 + b33 * b33 + 2 * off) / 6)
    safe = np.where(p > 0, p, 1)
    det = (
        b11 * (b22 * b33 - a23 * a23) - a12 * (a12 * b33 - a23 * a13) + a13 * (a12 * a23 - b22 * a13)
    ) / (safe * safe * safe)
    phi = np.arccos(np.clip(det / 2, -1, 1)) / 3
    largest = q + 2 * p * np.cos(phi)
    smallest = q + 2 * p * np.cos(phi + (2 * np.pi / 3))
    return largest, 3 * q - largest - smallest, smallest


def box_features(
    channels: Sequence[np.ndarray],
    voxel_size_zyx: Sequence[float],
    spec: FeatureSpec,
    core: tuple[slice, slice, slice],
) -> np.ndarray:
    """The features of the voxels in *core* of a padded box: *channels* are
    the box's scaled channels (:func:`scaled`), in *spec*'s order, and *core*
    the slices of the box whose features are wanted. Returns float32 of
    ``core shape + (spec.n_features,)``."""
    core_shape = tuple(s.stop - s.start for s in core)
    out = np.empty(core_shape + (spec.n_features,), dtype=np.float32)
    column = 0
    spacing = [float(v) for v in voxel_size_zyx]
    for channel in channels:
        channel = np.asarray(channel, dtype=np.float32)
        out[..., column] = channel[core]
        column += 1
        for scale in spec.scales_um:
            smooth = ndi.gaussian_filter(
                channel, sigma_voxels(scale, spacing), truncate=TRUNCATE, mode="reflect"
            )
            gz, gy, gx = (_difference(smooth, axis, spacing[axis]) for axis in range(3))
            out[..., column] = smooth[core]
            del smooth
            out[..., column + 1] = np.sqrt(gz[core] ** 2 + gy[core] ** 2 + gx[core] ** 2)
            hzz = _difference(gz, 0, spacing[0])[core]
            hzy = _difference(gz, 1, spacing[1])[core]
            hzx = _difference(gz, 2, spacing[2])[core]
            del gz
            hyy = _difference(gy, 1, spacing[1])[core]
            hyx = _difference(gy, 2, spacing[2])[core]
            del gy
            hxx = _difference(gx, 2, spacing[2])[core]
            del gx
            out[..., column + 2] = hzz + hyy + hxx
            for offset, eigenvalue in enumerate(symmetric_eigenvalues(hzz, hyy, hxx, hzy, hzx, hyx)):
                out[..., column + 3 + offset] = eigenvalue
            column += len(PER_SCALE)
    return out


def boxes(
    shape: Sequence[int],
    halo_zyx: Sequence[int],
    *,
    box_voxels: int = BOX_VOXELS,
    core_zyx: Sequence[int] | None = None,
) -> Iterator[tuple[tuple[slice, ...], tuple[slice, ...], tuple[slice, ...]]]:
    """``(padded, core in padded, core)`` slices tiling *shape*: the cores
    partition it, each padded by *halo_zyx* (clipped to the volume). Cores
    are *core_zyx*, or halved along their longest axis until a padded box
    holds no more than *box_voxels* -- but never thinner than twice the halo
    (or 16 voxels), where the margin would cost more than the core: a box
    may then hold more than *box_voxels*."""
    shape = tuple(int(n) for n in shape)
    halo = tuple(max(0, int(h)) for h in halo_zyx)
    if core_zyx is not None:
        core = [max(1, min(int(c), s)) for c, s in zip(core_zyx, shape)]
    else:
        core = list(shape)
        thinnest = [min(s, max(16, 2 * h)) for s, h in zip(shape, halo)]

        def padded(extents):
            return int(np.prod([min(s, c + 2 * h) for s, c, h in zip(shape, extents, halo)]))

        while padded(core) > box_voxels:
            shrinkable = [axis for axis in range(len(core)) if core[axis] > thinnest[axis]]
            if not shrinkable:
                break
            axis = max(shrinkable, key=lambda a: core[a])
            core[axis] = max(thinnest[axis], (core[axis] + 1) // 2)
    for z in range(0, shape[0], core[0]):
        for y in range(0, shape[1], core[1]):
            for x in range(0, shape[2], core[2]):
                starts = (z, y, x)
                core_slices = tuple(
                    slice(start, min(start + c, s)) for start, c, s in zip(starts, core, shape)
                )
                padded_slices = tuple(
                    slice(max(0, cs.start - h), min(s, cs.stop + h))
                    for cs, h, s in zip(core_slices, halo, shape)
                )
                inner = tuple(
                    slice(cs.start - ps.start, cs.stop - ps.start)
                    for cs, ps in zip(core_slices, padded_slices)
                )
                yield padded_slices, inner, core_slices


def features_of_box(
    volumes: Sequence[np.ndarray],
    ranges: Sequence[tuple[float, float]],
    voxel_size_zyx: Sequence[float],
    spec: FeatureSpec,
    padded: tuple[slice, ...],
    inner: tuple[slice, ...],
) -> np.ndarray:
    """The features of one box's core, read from the whole *volumes* (one
    per channel, canonical order, memory-mapped or not) on their *ranges*."""
    channels = [scaled(volume[padded], value_range) for volume, value_range in zip(volumes, ranges)]
    return box_features(channels, voxel_size_zyx, spec, inner)


def features_at(
    volumes: Sequence[np.ndarray],
    ranges: Sequence[tuple[float, float]],
    voxel_size_zyx: Sequence[float],
    spec: FeatureSpec,
    points_zyx: np.ndarray,
    *,
    box_voxels: int = BOX_VOXELS,
) -> np.ndarray:
    """The features at *points_zyx* (an ``(n, 3)`` integer array), float32
    ``(n, n_features)`` in the points' order: only the boxes holding a point
    are computed."""
    points = np.asarray(points_zyx, dtype=np.int64).reshape(-1, 3)
    out = np.empty((len(points), spec.n_features), dtype=np.float32)
    if not len(points):
        return out
    shape = np.shape(volumes[0])
    halo = feature_halo(spec, voxel_size_zyx)
    for padded, inner, core in boxes(shape, halo, box_voxels=box_voxels):
        inside = np.ones(len(points), dtype=bool)
        for axis, part in enumerate(core):
            inside &= (points[:, axis] >= part.start) & (points[:, axis] < part.stop)
        if not inside.any():
            continue
        features = features_of_box(volumes, ranges, voxel_size_zyx, spec, padded, inner)
        local = points[inside] - np.array([part.start for part in core])
        out[inside] = features[local[:, 0], local[:, 1], local[:, 2]]
    return out
