"""Vessel cross-sections: sampling them from a volume, and the models fitted to them.

Shared by every method that reads a vessel's width from the plane normal to
its centreline -- the raw image's own lumen
(:mod:`haemolynx.haemodynamics.raw_section`) and the endothelial wall round it
(:mod:`haemolynx.haemodynamics.endothelial`) -- so they sample, frame, blur
and model a section the same way.

The frame: ``t`` along the vessel, ``a`` across it in the image's y-x plane
(no z), ``b = t x a``. With ``a`` free of z, a PSF aligned to the image axes
projects onto ``(a, b)`` with no cross term, so a blurred model is two
one-dimensional Gaussian blurs.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .automated import _interpolate_centerline
from .edt_diameter import _centreline_tangents

__all__ = [
    "averaged_section",
    "lumen_image",
    "nearest_section",
    "projected_sigma",
    "psf_from_section_blurs",
    "ring_images",
    "section_axes",
]


#: Fewer edges than this are measured in this process: starting workers
#: (each imports numpy and scipy afresh on Windows) costs more than it saves.
MIN_EDGES_FOR_WORKERS = 64

_WORKER_VOLUME: np.ndarray | None = None


#: The most workers :func:`default_worker_count` gives: on a 20-thread
#: machine 500 real vessels' endothelial rings took 170 s in one process,
#: 41 s in 8 and 54 s in 19 -- past eight, starting workers and sharing the
#: physical cores costs more than the extra ones save.
MAX_DEFAULT_WORKERS = 8


def default_worker_count() -> int:
    """Up to :data:`MAX_DEFAULT_WORKERS`, leaving a core for the rest of the machine."""
    import os

    return max(1, min(MAX_DEFAULT_WORKERS, (os.cpu_count() or 2) - 1))


def _load_worker_volume(path: str, dtype: str, shape: tuple[int, ...]) -> None:
    global _WORKER_VOLUME
    _WORKER_VOLUME = np.memmap(path, dtype=np.dtype(dtype), mode="r", shape=shape)


_NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def _run_on_worker_volume(task_and_payload):
    task, payload = task_and_payload
    return task(_WORKER_VOLUME, payload)


def run_edge_tasks(task, payloads, *, volume: np.ndarray, workers: int = 1, memmap_directory=None):
    """``[task(volume, payload) for payload in payloads]``, spread over
    *workers* processes when there are enough payloads to repay starting them.

    Each edge's measurement depends only on its own payload and the volume,
    so the results are the same, in the same order, however many processes
    compute them. The workers read the volume from one temporary memory-mapped
    file rather than each receiving a copy, so memory stays one volume's
    worth. *task* must be a module-level function (it is sent to the workers
    by name). Falls back to this process, with a warning, if workers cannot
    be started.
    """
    payloads = list(payloads)
    if workers <= 1 or len(payloads) < MIN_EDGES_FOR_WORKERS:
        return [task(volume, payload) for payload in payloads]
    import logging
    from concurrent.futures import ProcessPoolExecutor

    from haemolynx.preprocessing.memmap_support import new_memmap_array, release_memmap_array

    import os

    shared = new_memmap_array(volume.shape, np.float32, directory=memmap_directory)
    # Workers read these as they load numpy: one native thread each, since
    # the workers already fill the cores and a BLAS/OpenMP team in each would
    # oversubscribe them.
    single_threaded = {name: "1" for name in _NATIVE_THREAD_VARIABLES}
    saved = {name: os.environ.get(name) for name in single_threaded}
    try:
        shared[...] = volume
        shared.flush()
        chunk = max(1, len(payloads) // (4 * workers))
        os.environ.update(single_threaded)
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_load_worker_volume,
                initargs=(str(shared.filename), "float32", tuple(shared.shape)),
            ) as pool:
                return list(pool.map(_run_on_worker_volume, [(task, p) for p in payloads], chunksize=chunk))
        except (OSError, RuntimeError) as error:  # e.g. no process spawning here
            logging.getLogger(__name__).warning(
                "Could not start %d worker processes (%s); measuring in this one.", workers, error
            )
            return [task(volume, payload) for payload in payloads]
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        release_memmap_array(shared)


def section_axes(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit tangent ``t`` and the section's axes ``a`` (in the y-x plane) and
    ``b = t x a``."""
    t = np.asarray(tangent, dtype=float)
    t = t / max(float(np.linalg.norm(t)), 1e-12)
    a = np.cross(t, np.array([1.0, 0.0, 0.0]))  # axis 0 is z
    if np.linalg.norm(a) < 1e-6:
        a = np.array([0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(t, a)
    b /= np.linalg.norm(b)
    return t, a, b


def projected_sigma(axis: np.ndarray, psf_sigma_zyx) -> float:
    """The blur (Gaussian sigma, um) a PSF of *psf_sigma_zyx* has along *axis*."""
    sz, sy, sx = (float(v) for v in psf_sigma_zyx)
    return float(np.sqrt((axis[0] * sz) ** 2 + (axis[1] * sy) ** 2 + (axis[2] * sx) ** 2))


def averaged_section(
    volume: np.ndarray,
    poly: np.ndarray,
    s: np.ndarray,
    total_len: float,
    at: float,
    voxel_size_zyx,
    offsets: np.ndarray,
    average_um: float,
) -> np.ndarray:
    """*volume* in the plane normal to the centreline at arc length *at*,
    averaged over *average_um* of the vessel along its curve -- each plane in
    its own frame, so a curved vessel stays centred. Read in one call for
    every plane. NaN outside the volume."""
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    half = 0.5 * max(float(average_um), 0.0)
    step = 0.5 * float(np.min(spacing))
    along = np.arange(max(0.0, at - half), min(total_len, at + half) + 1e-9, step)
    if along.size == 0:
        along = np.array([at])
    points = _interpolate_centerline(poly, s, along)
    tangents = _centreline_tangents(poly, s, total_len, along)
    frames = [section_axes(tangent) for tangent in tangents]
    a = np.array([frame[1] for frame in frames])
    b = np.array([frame[2] for frame in frames])
    planes = (
        points[:, None, None, :]
        + offsets[None, :, None, None] * a[:, None, None, :]
        + offsets[None, None, :, None] * b[:, None, None, :]
    )
    # Float output: map_coordinates otherwise returns the volume's own dtype,
    # and a uint16 stack would come back truncated, with no NaN to mark where
    # the plane leaves it.
    values = map_coordinates(
        volume, (planes / spacing).reshape(-1, 3).T, order=1, mode="constant",
        cval=np.nan, output=np.float64,
    )
    return values.reshape(len(points), len(offsets), len(offsets)).mean(axis=0)


def nearest_section(volume, point, a, b, offsets, voxel_size_zyx) -> np.ndarray:
    """Whether the voxel nearest each point of the plane through *point*
    spanned by *a* and *b* is non-zero; False outside the volume. Reads only
    those voxels, so a memory-mapped mask stays on disk."""
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    plane = point + offsets[:, None, None] * a + offsets[None, :, None] * b
    nearest = np.rint((plane / spacing).reshape(-1, 3)).astype(np.intp)
    inside = np.all((nearest >= 0) & (nearest < np.asarray(volume.shape)), axis=1)
    values = np.zeros(len(nearest), dtype=bool)
    hits = nearest[inside]
    values[inside] = np.asarray(volume[hits[:, 0], hits[:, 1], hits[:, 2]]) != 0
    return values.reshape(len(offsets), len(offsets))


def _ellipse_cover(xs, ys, cx, cy, ra, rb, step) -> np.ndarray:
    """1 inside the ellipse, 0 outside, soft over one grid step at its edge."""
    x = xs[:, None] - cx
    y = ys[None, :] - cy
    radius = np.sqrt((x / ra) ** 2 + (y / rb) ** 2)
    signed = (radius - 1.0) * np.sqrt(ra * rb)
    return np.clip(0.5 - signed / step, 0.0, 1.0)


def _blur(cover, sigma_a, sigma_b, step) -> np.ndarray:
    return gaussian_filter(
        cover, (max(sigma_a / step, 1e-3), max(sigma_b / step, 1e-3)), mode="constant"
    )


def lumen_image(xs, ys, cx, cy, ra, rb, sigma_a, sigma_b, step) -> np.ndarray:
    """A unit-height elliptical lumen (semi-axes *ra*, *rb* along the
    section's a and b) on the grid *xs* x *ys*, seen through a Gaussian blur
    of *sigma_a*, *sigma_b* (um). The grid may be a crop of the plane: the
    lumen must lie inside it, or its blur is cut off at the crop's edge."""
    return _blur(_ellipse_cover(xs, ys, cx, cy, ra, rb, step), sigma_a, sigma_b, step)


def ring_images(xs, ys, cx, cy, ra, rb, wall, sigma_a, sigma_b, step) -> tuple[np.ndarray, np.ndarray]:
    """``(lumen, wall)``: a unit elliptical lumen of semi-axes *ra*, *rb* and
    the unit ring of thickness *wall* round it, each seen through the blur.
    A vessel stained on its endothelium is ``background + L * lumen + W * wall``."""
    inner = _ellipse_cover(xs, ys, cx, cy, ra, rb, step)
    outer = _ellipse_cover(xs, ys, cx, cy, ra + wall, rb + wall, step)
    return _blur(inner, sigma_a, sigma_b, step), _blur(outer - inner, sigma_a, sigma_b, step)


def psf_from_section_blurs(sigma_a, sigma_b, other_axis_z) -> tuple[float, float]:
    """``(sigma_xy, sigma_z)`` from sections' fitted blurs.

    The across axis ``a`` lies in the y-x plane, so its blur is sigma_xy; the
    other axis's is ``sigma_xy^2 (1 - b_z^2) + sigma_z^2 b_z^2``. A robust
    (iteratively down-weighted) least-squares fit over every section gives
    both."""
    sa = np.asarray(sigma_a, dtype=float)
    sb = np.asarray(sigma_b, dtype=float)
    bz = np.asarray(other_axis_z, dtype=float)
    design = np.concatenate(
        [np.c_[np.ones_like(sa), np.zeros_like(sa)], np.c_[1.0 - bz ** 2, bz ** 2]]
    )
    target = np.concatenate([sa ** 2, sb ** 2])
    weights = np.ones_like(target)
    coef = np.zeros(2)
    for _iteration in range(10):
        coef, *_ = np.linalg.lstsq(design * weights[:, None], target * weights, rcond=None)
        residual = target - design @ coef
        scale = 1.4826 * float(np.median(np.abs(residual))) + 1e-12
        weights = 1.0 / np.maximum(1.0, np.abs(residual) / (1.5 * scale))
    sigma_xy, sigma_z = (float(np.sqrt(max(c, 1e-6))) for c in coef)
    return sigma_xy, sigma_z
