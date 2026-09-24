"""Input axis-order handling and canonical (z, y, x) conventions.

HaemoLynx works internally in a canonical ``(z, y, x)`` array layout:

* array axis 0 is ``z`` — the stack axis, i.e. the axis that overlays and
  maximum-intensity projections are projected *through*,
* array axis 1 is ``y``,
* array axis 2 is ``x``.

Node ``pos`` attributes, edge ``voxels`` paths and every per-array-axis spacing
vector follow that same ``(z, y, x)`` order. Voxel sizes read from file
metadata, by contrast, are physically labelled and are therefore carried as
``(x, y, z)``; :func:`voxel_size_zyx_from_xyz` converts between the two. Mixing
the two orders silently mis-scales anisotropic data, so conversion happens once,
at the loader boundary.

Input files do not always arrive in ``(z, y, x)``. Loaders accept an
``axis_order`` string naming what each *array* axis means (``"xyz"`` for an
(x, y, z) volume, ``"yzx"`` for a (y, z, x) volume, and so on) and transpose the
volume to the canonical order on load. That is what makes "which axis is z" —
the axis projected through — user-selectable.
"""
from __future__ import annotations

import numpy as np

CANONICAL_AXIS_ORDER = "zyx"
VALID_AXIS_ORDERS = (
    "zyx",
    "zxy",
    "yzx",
    "yxz",
    "xzy",
    "xyz",
)


def normalize_axis_order(axis_order: str, *, label: str = "axis_order") -> str:
    """Validate an axis-order string and return it lowercased.

    The string names the meaning of each array axis in order, so ``"zyx"``
    describes a volume indexed ``volume[z, y, x]``.
    """
    if not isinstance(axis_order, str):
        raise ValueError(
            f"{label} must be a string permutation of 'xyz', got {axis_order!r}."
        )
    normalized = axis_order.strip().lower()
    if sorted(normalized) != ["x", "y", "z"]:
        raise ValueError(
            f"{label} must be a permutation of 'xyz' (one of {', '.join(VALID_AXIS_ORDERS)}), "
            f"got {axis_order!r}."
        )
    return normalized


def axis_order_transpose(axis_order: str) -> tuple[int, int, int]:
    """Return the ``np.transpose`` axes that map *axis_order* to ``(z, y, x)``."""
    normalized = normalize_axis_order(axis_order)
    return tuple(normalized.index(axis) for axis in CANONICAL_AXIS_ORDER)


def apply_axis_order(
    volume: np.ndarray,
    axis_order: str = CANONICAL_AXIS_ORDER,
    *,
    use_memmap: bool = False,
    memmap_directory=None,
) -> np.ndarray:
    """Transpose *volume* from *axis_order* into the canonical ``(z, y, x)`` order.

    ``axis_order="zyx"`` is a no-op. Other orders return a contiguous copy so
    downstream skeletonization and distance transforms are not slowed by a
    non-contiguous view.

    Uses ``asanyarray``, not ``asarray``, for the no-op case specifically so
    it is a genuine no-op even for an ``ndarray`` subclass -- a
    ``numpy.memmap`` loaded with ``use_memmap_loading`` would otherwise come
    back demoted to a plain in-RAM-looking ``ndarray`` (same underlying
    disk-backed buffer, since it is still a view rather than a copy, but the
    wrong type for anything downstream that checks ``isinstance(x,
    np.memmap)`` to decide whether it owns a backing file to release).

    *use_memmap*, when True, writes that contiguous copy one output slice at
    a time into a new memmap in *memmap_directory* instead of
    ``np.ascontiguousarray``, which is a whole-volume plain-RAM copy no
    matter how *volume* itself is stored. Slower -- each output slice
    gathers from across the whole source -- but never holds more than one
    slice.
    """
    normalized = normalize_axis_order(axis_order)
    arr = np.asanyarray(volume)
    if normalized == CANONICAL_AXIS_ORDER:
        return arr
    if arr.ndim != 3:
        raise ValueError(
            f"axis_order={axis_order!r} requires a 3D volume, got shape {arr.shape}."
        )
    transposed = np.transpose(arr, axis_order_transpose(normalized))
    if not use_memmap:
        return np.ascontiguousarray(transposed)
    from haemolynx.preprocessing.memmap_support import new_memmap_array

    result = new_memmap_array(transposed.shape, transposed.dtype, directory=memmap_directory)
    for index in range(transposed.shape[0]):
        result[index] = transposed[index]
    return result


def voxel_size_zyx_from_xyz(
    voxel_size_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert a physical ``(x, y, z)`` voxel size to canonical array order ``(z, y, x)``.

    Use this wherever a voxel size read from image metadata is handed to code
    that scales array indices — graph construction, distance transforms, FWHM
    sampling. Passing ``(x, y, z)`` straight through mis-scales anisotropic
    volumes by swapping the z and x spacings.
    """
    arr = np.asarray(voxel_size_xyz, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError(
            f"voxel_size_xyz must have length 3, got {voxel_size_xyz}."
        )
    return (float(arr[2]), float(arr[1]), float(arr[0]))


def voxel_size_xyz_from_zyx(
    voxel_size_zyx: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert a canonical array-order ``(z, y, x)`` voxel size back to ``(x, y, z)``."""
    arr = np.asarray(voxel_size_zyx, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError(
            f"voxel_size_zyx must have length 3, got {voxel_size_zyx}."
        )
    return (float(arr[2]), float(arr[1]), float(arr[0]))
