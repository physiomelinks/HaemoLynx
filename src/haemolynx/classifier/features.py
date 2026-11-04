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
from typing import Iterator, Mapping, Sequence

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
    *coarse*: each wide scale is smoothed on a grid coarser by
    :func:`coarse_factors` and interpolated back (:func:`box_features`), at a
    fraction of the cost; a classifier saved before records none and reads
    every scale on the image's own grid, as it was trained.
    """

    scales_um: tuple[float, ...] = DEFAULT_SCALES_UM
    channels: tuple[int | None, ...] = (None,)
    coarse: bool = False

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
        object.__setattr__(self, "coarse", bool(self.coarse))

    @property
    def n_features(self) -> int:
        return len(self.channels) * (1 + len(PER_SCALE) * len(self.scales_um))

    def names(self) -> list[str]:
        """Each feature's name, in the order a feature array holds them."""
        names = []
        for channel in self.channels:
            prefix = "image" if channel is None else f"C{channel + 1}"
            names.append(f"{prefix} intensity")
            grid = " (coarse grid)" if self.coarse else ""
            for scale in self.scales_um:
                names += [f"{prefix} {kind} {scale:g} um{grid}" for kind in PER_SCALE]
        return names

    def to_record(self) -> dict:
        return {"scales_um": list(self.scales_um), "channels": list(self.channels), "coarse": self.coarse}

    @classmethod
    def from_record(cls, record: dict) -> "FeatureSpec":
        return cls(
            scales_um=tuple(record["scales_um"]),
            channels=tuple(record.get("channels", [None])),
            coarse=bool(record.get("coarse", False)),
        )


def sigma_voxels(scale_um: float, voxel_size_zyx: Sequence[float]) -> tuple[float, float, float]:
    """A scale in microns as a Gaussian sigma in voxels along z, y and x."""
    return tuple(float(scale_um) / float(size) for size in voxel_size_zyx)  # type: ignore[return-value]


#: On a coarse grid a scale keeps at least this sigma, in coarse voxels.
COARSE_SIGMA_VOXELS = 2.0


def coarse_factors(spec: FeatureSpec, scale_um: float, voxel_size_zyx: Sequence[float]) -> tuple[int, int, int]:
    """How many of the image's voxels one coarse voxel of *scale_um* spans
    along z, y and x: the sigma in voxels over :data:`COARSE_SIGMA_VOXELS`,
    rounded down, at least 1; all 1 unless *spec* is coarse."""
    if not spec.coarse:
        return (1, 1, 1)
    return tuple(  # type: ignore[return-value]
        max(1, int(sigma // COARSE_SIGMA_VOXELS)) for sigma in sigma_voxels(scale_um, voxel_size_zyx)
    )


def scale_halo(spec: FeatureSpec, scale_um: float, voxel_size_zyx: Sequence[float]) -> tuple[int, int, int]:
    """The margin (voxels, per axis) one scale's features need to be the
    whole volume's: its Gaussian's reach (as ``scipy.ndimage`` rounds it)
    and two more for the second differences -- on a coarse grid, in coarse
    voxels, two more for the interpolation and the grid's alignment, times
    the factor."""
    factors = coarse_factors(spec, scale_um, voxel_size_zyx)
    sigmas = sigma_voxels(scale_um, voxel_size_zyx)
    if factors == (1, 1, 1):
        return tuple(int(TRUNCATE * sigma + 0.5) + 2 for sigma in sigmas)  # type: ignore[return-value]
    return tuple(  # type: ignore[return-value]
        (int(TRUNCATE * sigma / f + 0.5) + 4) * f for sigma, f in zip(sigmas, factors)
    )


def feature_halo(spec: FeatureSpec, voxel_size_zyx: Sequence[float]) -> tuple[int, int, int]:
    """The margin (voxels, per axis) a box needs for its core's features to
    be the whole volume's: the largest a scale on the image's own grid needs
    (:func:`scale_halo`). A scale on a coarse grid reads its own, wider,
    region from the image (:func:`coarse_reach`) and is not counted: the box
    pays for its margin only in a block average."""
    halos = [
        scale_halo(spec, scale, voxel_size_zyx)
        for scale in spec.scales_um
        if coarse_factors(spec, scale, voxel_size_zyx) == (1, 1, 1)
    ]
    if not halos:
        return (2, 2, 2)
    return tuple(max(h[axis] for h in halos) for axis in range(3))  # type: ignore[return-value]


def coarse_reach(spec: FeatureSpec, voxel_size_zyx: Sequence[float]) -> tuple[int, int, int]:
    """The margin (voxels, per axis) round a box the coarse scales read from
    the image: what a box's pages must reach, with :func:`feature_halo`."""
    reach = [scale_halo(spec, scale, voxel_size_zyx) for scale in spec.scales_um]
    return tuple(max(h[axis] for h in reach) for axis in range(3))  # type: ignore[return-value]


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


def _scale_features(smooth: np.ndarray, spacing, core, want: np.ndarray) -> list:
    """A scale's six features at *core* of its smoothed region (None where
    not *want*-ed): the smoothed intensity, its gradient magnitude, its
    Laplacian and the Hessian's eigenvalues, largest first."""
    found: list = [None] * len(PER_SCALE)
    if want[0]:
        found[0] = smooth[core]
    if not want[1:].any():
        return found
    gz, gy, gx = (_difference(smooth, axis, spacing[axis]) for axis in range(3))
    if want[1]:
        found[1] = np.sqrt(gz[core] ** 2 + gy[core] ** 2 + gx[core] ** 2)
    if want[2:].any():
        eigen = bool(want[3:].any())
        hzz = _difference(gz, 0, spacing[0])[core]
        hzy = _difference(gz, 1, spacing[1])[core] if eigen else None
        hzx = _difference(gz, 2, spacing[2])[core] if eigen else None
        del gz
        hyy = _difference(gy, 1, spacing[1])[core]
        hyx = _difference(gy, 2, spacing[2])[core] if eigen else None
        del gy
        hxx = _difference(gx, 2, spacing[2])[core]
        del gx
        if want[2]:
            found[2] = hzz + hyy + hxx
        if eigen:
            for offset, eigenvalue in enumerate(symmetric_eigenvalues(hzz, hyy, hxx, hzy, hzx, hyx)):
                if want[3 + offset]:
                    found[3 + offset] = eigenvalue
    return found


def _block_mean(values: np.ndarray, factors: Sequence[int]) -> np.ndarray:
    """The mean of each block of *factors* voxels (fewer at the far end,
    where the region stops at the volume's face), float32, axis by axis."""
    out = np.asarray(values, dtype=np.float32)
    for axis, f in enumerate(factors):
        if f == 1:
            continue
        n = out.shape[axis]
        blocks = -(-n // f)
        pad = [(0, 0)] * out.ndim
        pad[axis] = (0, blocks * f - n)
        padded = np.pad(out, pad)
        shape = list(padded.shape)
        shape[axis:axis + 1] = [blocks, f]
        sums = padded.reshape(shape).sum(axis=axis + 1, dtype=np.float32)
        counts = np.full(blocks, f, dtype=np.float32)
        counts[-1] = n - (blocks - 1) * f
        view = [1] * out.ndim
        view[axis] = blocks
        out = sums / counts.reshape(view)
    return out


def _interpolate(values: np.ndarray, starts: Sequence[int], factors: Sequence[int], first: Sequence[int],
                 core_global: Sequence[slice], coarse_size: Sequence[int]) -> np.ndarray:
    """*values* (coarse voxels from global coarse index *first*) linearly
    interpolated, axis by axis, at the image voxels *core_global*: an image
    voxel ``i`` sits at coarse position ``(i + 0.5) / f - 0.5``, held at the
    first and last coarse voxels."""
    out = values
    for axis, (f, part, nc) in enumerate(zip(factors, core_global, coarse_size)):
        if f == 1:
            index = np.arange(part.start, part.stop) - first[axis]
            out = np.take(out, index, axis=axis)
            continue
        position = (np.arange(part.start, part.stop, dtype=np.float64) + 0.5) / f - 0.5
        lower = np.floor(position).astype(np.int64)
        weight = (position - lower).astype(np.float32)
        weight[lower < 0] = 0
        lower = np.clip(lower, 0, nc - 1)
        upper = np.minimum(lower + 1, nc - 1)
        view = [1] * out.ndim
        view[axis] = len(weight)
        weight = weight.reshape(view)
        low = np.take(out, lower - first[axis], axis=axis)
        high = np.take(out, upper - first[axis], axis=axis)
        out = low + weight * (high - low)
    return out


def _coarse_scale_features(channel, origin, volume_shape, core, spacing, sigmas, factors, want, source=None) -> list:
    """A scale's features at *core* (slices of the padded *channel*, whose
    first voxel is the image's *origin*), computed on a grid *factors*
    coarser: block means aligned to the whole image, smoothed and
    differentiated there, interpolated back to the image's voxels. A voxel's
    values depend only on the image, never on the box. *source* --
    ``(volume, range)`` -- is where the region past the box is read from."""
    core_global = [slice(o + c.start, o + c.stop) for o, c in zip(origin, core)]
    coarse_size = [-(-n // f) for n, f in zip(volume_shape, factors)]
    halo = [int(TRUNCATE * s / f + 0.5) + 2 for s, f in zip(sigmas, factors)]
    first, last = [], []
    for part, f, nc, h in zip(core_global, factors, coarse_size, halo):
        low = int(np.floor((part.start + 0.5) / f - 0.5))
        high = int(np.floor((part.stop - 1 + 0.5) / f - 0.5)) + 1
        first.append(max(0, min(low, nc - 1) - h))
        last.append(min(nc, max(high, 0) + 1 + h))
    region = tuple(
        slice(lo * f - o, min(n, hi * f) - o)
        for lo, hi, f, n, o in zip(first, last, factors, volume_shape, origin)
    )
    if all(r.start >= 0 and r.stop <= s for r, s in zip(region, channel.shape)):
        values = channel[region]
    elif source is not None:
        volume, value_range = source
        values = scaled(volume[tuple(slice(r.start + o, r.stop + o) for r, o in zip(region, origin))], value_range)
    else:
        raise ValueError("The box reaches past its channel and no volume was given to read the rest from.")
    coarse = _block_mean(values, factors)
    smooth = ndi.gaussian_filter(
        coarse, tuple(s / f for s, f in zip(sigmas, factors)), truncate=TRUNCATE, mode="reflect"
    )
    whole = tuple(slice(0, n) for n in smooth.shape)
    found = _scale_features(smooth, [sp * f for sp, f in zip(spacing, factors)], whole, want)
    return [
        None if values is None else _interpolate(values, region, factors, first, core_global, coarse_size)
        for values in found
    ]


@dataclass
class CoarseFeatures:
    """One coarse scale's features over a region, on its coarse grid: what
    the boxes inside the region interpolate theirs from."""

    first: list  # the coarse index the arrays start at, per axis
    factors: tuple
    coarse_size: list
    found: list  # per feature of the scale: a coarse array, or None


def coarse_scale_features(source, voxel_size_zyx, spec, scale, region, volume_shape, want) -> CoarseFeatures:
    """The coarse features of *scale* for the image voxels of *region*
    (global slices): read from *source* -- ``(volume, range)`` -- with the
    coarse margin, block-averaged, smoothed and differentiated, and kept for
    the coarse voxels *region* interpolates from. The values a voxel gets
    from them are those :func:`box_features` gives it computing alone."""
    spacing = [float(v) for v in voxel_size_zyx]
    factors = coarse_factors(spec, scale, spacing)
    sigmas = sigma_voxels(scale, spacing)
    coarse_size = [-(-n // f) for n, f in zip(volume_shape, factors)]
    halo = [int(TRUNCATE * s / f + 0.5) + 2 for s, f in zip(sigmas, factors)]
    first, last, keep_first, keep_last = [], [], [], []
    for part, f, nc, h in zip(region, factors, coarse_size, halo):
        low = min(max(int(np.floor((part.start + 0.5) / f - 0.5)), 0), nc - 1)
        high = min(max(int(np.floor((part.stop - 1 + 0.5) / f - 0.5)) + 1, 0), nc - 1)
        first.append(max(0, low - h))
        last.append(min(nc, high + 1 + h))
        keep_first.append(low)
        keep_last.append(high + 1)
    read = tuple(slice(lo * f, min(n, hi * f)) for lo, hi, f, n in zip(first, last, factors, volume_shape))
    volume, value_range = source
    coarse = _block_mean(scaled(volume[read], value_range), factors)
    smooth = ndi.gaussian_filter(
        coarse, tuple(s / f for s, f in zip(sigmas, factors)), truncate=TRUNCATE, mode="reflect"
    )
    keep = tuple(slice(k0 - f0, k1 - f0) for k0, k1, f0 in zip(keep_first, keep_last, first))
    found = _scale_features(smooth, [sp * f for sp, f in zip(spacing, factors)], keep, want)
    return CoarseFeatures(first=keep_first, factors=factors, coarse_size=coarse_size, found=found)


def interpolate_coarse(features: CoarseFeatures, core_global) -> list:
    """*features*' values at the image voxels *core_global*."""
    return [
        None if values is None
        else _interpolate(values, None, features.factors, features.first, core_global, features.coarse_size)
        for values in features.found
    ]


def coarse_context(volumes, ranges, voxel_size_zyx, spec: FeatureSpec, region, needed=None) -> dict:
    """Every coarse scale's features for the image voxels of *region*,
    ``{(channel number, scale): CoarseFeatures}``: computed once for a large
    region and shared by the boxes inside it, so none of them reads or
    averages the coarse margin itself."""
    if needed is None:
        needed = np.ones(spec.n_features, dtype=bool)
    needed = np.asarray(needed, dtype=bool)
    volume_shape = tuple(int(n) for n in np.shape(volumes[0]))
    width = 1 + len(PER_SCALE) * len(spec.scales_um)
    context = {}
    for number, (volume, value_range) in enumerate(zip(volumes, ranges)):
        for index, scale in enumerate(spec.scales_um):
            if coarse_factors(spec, scale, voxel_size_zyx) == (1, 1, 1):
                continue
            start = number * width + 1 + index * len(PER_SCALE)
            want = needed[start:start + len(PER_SCALE)]
            if want.any():
                context[(number, scale)] = coarse_scale_features(
                    (volume, value_range), voxel_size_zyx, spec, scale, region, volume_shape, want
                )
    return context


def has_coarse_scales(spec: FeatureSpec, voxel_size_zyx: Sequence[float]) -> bool:
    return any(coarse_factors(spec, scale, voxel_size_zyx) != (1, 1, 1) for scale in spec.scales_um)


def box_features(
    channels: Sequence[np.ndarray],
    voxel_size_zyx: Sequence[float],
    spec: FeatureSpec,
    core: tuple[slice, slice, slice],
    *,
    needed: np.ndarray | None = None,
    origin: Sequence[int] = (0, 0, 0),
    volume_shape: Sequence[int] | None = None,
    sources: Sequence | None = None,
    coarse: Mapping | None = None,
) -> np.ndarray:
    """The features of the voxels in *core* of a padded box: *channels* are
    the box's scaled channels (:func:`scaled`), in *spec*'s order, and *core*
    the slices of the box whose features are wanted. Returns float32 of
    ``core shape + (spec.n_features,)``.

    *needed* (a bool per feature) skips the work for features no tree reads:
    a scale none of whose features is read is not smoothed, the Hessian is
    not taken where neither its trace nor its eigenvalues are read. A feature
    computed is the same value either way; one skipped is 0.

    Each scale is computed over its own margin round the core
    (:func:`scale_halo`), not the widest scale's. A coarse *spec*'s wide
    scales are computed on their coarse grid (:func:`coarse_factors`), which
    needs the box's place in the image: *origin*, its first voxel, and
    *volume_shape* (the box's own shape when not given), and *sources* --
    each channel's ``(volume, range)`` -- to read its region past the box.
    *coarse* (:func:`coarse_context` of a region holding the box) gives the
    coarse scales' values ready to interpolate, the same values."""
    core_shape = tuple(s.stop - s.start for s in core)
    if needed is None:
        needed = np.ones(spec.n_features, dtype=bool)
        out = np.empty(core_shape + (spec.n_features,), dtype=np.float32)
    else:
        needed = np.asarray(needed, dtype=bool)
        out = np.zeros(core_shape + (spec.n_features,), dtype=np.float32)
    column = 0
    spacing = [float(v) for v in voxel_size_zyx]
    for number, channel in enumerate(channels):
        channel_needed = needed[column: column + 1 + len(PER_SCALE) * len(spec.scales_um)]
        if not channel_needed.any():
            column += len(channel_needed)
            continue
        channel = np.asarray(channel, dtype=np.float32)
        if needed[column]:
            out[..., column] = channel[core]
        column += 1
        for scale in spec.scales_um:
            want = needed[column: column + len(PER_SCALE)]
            if not want.any():
                column += len(PER_SCALE)
                continue
            factors = coarse_factors(spec, scale, spacing)
            if factors == (1, 1, 1):
                # Only this scale's own margin: the same values, less work.
                halo = scale_halo(spec, scale, spacing)
                region = tuple(
                    slice(max(0, c.start - h), min(n, c.stop + h)) for c, h, n in zip(core, halo, channel.shape)
                )
                inner = tuple(slice(c.start - r.start, c.stop - r.start) for c, r in zip(core, region))
                smooth = ndi.gaussian_filter(
                    channel[region], sigma_voxels(scale, spacing), truncate=TRUNCATE, mode="reflect"
                )
                found = _scale_features(smooth, spacing, inner, want)
                del smooth
            elif coarse is not None and (number, scale) in coarse:
                core_global = [slice(o + c.start, o + c.stop) for o, c in zip(origin, core)]
                found = interpolate_coarse(coarse[(number, scale)], core_global)
            else:
                found = _coarse_scale_features(
                    channel, origin, volume_shape or channel.shape, core, spacing,
                    sigma_voxels(scale, spacing), factors, want,
                    source=None if sources is None else sources[number],
                )
            for offset, values in enumerate(found):
                if values is not None:
                    out[..., column + offset] = values
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
    *,
    needed: np.ndarray | None = None,
    coarse: Mapping | None = None,
) -> np.ndarray:
    """The features of one box's core, read from the whole *volumes* (one
    per channel, canonical order, in memory, memory-mapped or read lazily)
    on their *ranges*. A channel none of whose features is *needed* is not
    read."""
    channels = []
    width = 1 + len(PER_SCALE) * len(spec.scales_um)
    for index, (volume, value_range) in enumerate(zip(volumes, ranges)):
        if needed is not None and not np.asarray(needed)[index * width:(index + 1) * width].any():
            channels.append(None)
        else:
            channels.append(scaled(volume[padded], value_range))
    return box_features(
        channels, voxel_size_zyx, spec, inner, needed=needed,
        origin=tuple(p.start for p in padded), volume_shape=tuple(int(n) for n in np.shape(volumes[0])),
        sources=list(zip(volumes, ranges)), coarse=coarse,
    )


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
