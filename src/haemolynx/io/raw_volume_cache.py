"""Reuse ``use_memmap_loading``'s decompressed raw volume across pipeline runs.

Every other memmap-backed buffer in this codebase (``fill_binary_holes``'s
labelled background, the EDT transforms in
:mod:`haemolynx.preprocessing.segmentation_cleanup`, ...) is transient scratch,
released the moment its own step finishes -- correctly, because those depend
on settings that can change between runs (segmentation-cleanup toggles,
bridge radii, ...) and must always be recomputed. The raw loaded volume this
module caches is different: it depends on nothing but the source file's own
bytes and ``image_axis_order``, so once a run has decompressed it into
``memmap_directory``, re-reading and re-decompressing the same file for a
later run is wasted work -- exactly the "prevent rebuilding" this exists for.

Only this one array is reused. The skeleton, the cleaned mask, the graph and
everything after it are always rebuilt fresh from whatever ``image`` this
returns, precisely as if it had just been loaded -- turning this on cannot
make a run see stale results for anything a *setting* controls, only skip
the one read/decompress step that no setting can change the outcome of.

Restricted to ``image_axis_order == "zyx"`` (the canonical, default order):
for any other order, :func:`haemolynx.io.axis_order.apply_axis_order`
transposes the volume into a second, fresh memmap, whose shape no longer
matches the bytes as written to disk in the file's own native axis order.
Caching that would silently reopen a later run's file with the wrong axis
mapping. A non-canonical order simply always misses and falls through to an
ordinary (still memmap-backed) load.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .axis_order import CANONICAL_AXIS_ORDER
from .load_2d import load_image_with_voxel_size_2d_aware

logger = logging.getLogger(__name__)

#: Distinguishes a persistent cache entry from the "haemolynx_*.memmap"
#: random-named scratch files this run's other transient buffers create in
#: the same directory, so neither is ever mistaken for the other.
_CACHE_PREFIX = "haemolynx_raw_cache_"


def _cache_key(resolved_path: Path, stat_result, axis_order: str) -> str:
    payload = f"{resolved_path}|{stat_result.st_mtime_ns}|{stat_result.st_size}|{axis_order}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _cache_file_paths(directory: Path, key: str) -> tuple[Path, Path]:
    base = directory / f"{_CACHE_PREFIX}{key}"
    return base.with_suffix(".memmap"), base.with_suffix(".json")


def _read_cache_entry(memmap_path: Path, sidecar_path: Path):
    """``(image, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status)``
    or ``None`` for any entry that cannot be trusted -- missing, truncated,
    or written in a format this version does not recognise. A cache is an
    optimisation, never a correctness requirement, so every failure here is
    treated as a miss rather than raised.
    """
    try:
        meta = json.loads(sidecar_path.read_text())
        shape = tuple(int(v) for v in meta["shape"])
        dtype = np.dtype(meta["dtype"])
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if memmap_path.stat().st_size != expected_bytes:
            return None
        image = np.memmap(memmap_path, dtype=dtype, mode="r", shape=shape)
        return (
            image,
            float(meta["voxel_size_x"]),
            float(meta["voxel_size_y"]),
            float(meta["voxel_size_z"]),
            meta["voxel_meta_status"],
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _write_sidecar(
    sidecar_path: Path,
    image: np.ndarray,
    voxel_size_x: float,
    voxel_size_y: float,
    voxel_size_z: float,
    voxel_meta_status: dict[str, object],
) -> None:
    meta = {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "voxel_size_x": float(voxel_size_x),
        "voxel_size_y": float(voxel_size_y),
        "voxel_size_z": float(voxel_size_z),
        "voxel_meta_status": voxel_meta_status,
    }
    sidecar_path.write_text(json.dumps(meta))


def load_raw_volume_reusing_cache(
    filepath: str,
    *,
    input_format: str,
    axis_order: str,
    memmap_directory: str | Path,
    h5_dataset_name: str | None = None,
) -> tuple[np.ndarray, float, float, float, dict[str, object]]:
    """Like :func:`load_image_with_voxel_size_2d_aware` with ``use_memmap=True``,
    but reuses a previous run's own decompressed copy from *memmap_directory*
    when the source file and *axis_order* have not changed since, instead of
    reading *filepath* again.

    Invalidated by anything that changes what the returned array actually
    contains: the resolved source path, its mtime and size (a changed file --
    including ilastik simply re-running and overwriting its own output --
    always misses and is rebuilt), and *axis_order*. See the module
    docstring for why a non-canonical *axis_order* always misses too.
    """
    if axis_order != CANONICAL_AXIS_ORDER:
        logger.debug(
            "Not reusing a cached raw volume: image_axis_order=%r is not "
            "canonical, so it is transposed into a fresh memmap each run "
            "instead of cached.",
            axis_order,
        )
        return load_image_with_voxel_size_2d_aware(
            filepath,
            input_format=input_format,
            axis_order=axis_order,
            h5_dataset_name=h5_dataset_name,
            use_memmap=True,
            memmap_directory=memmap_directory,
        )

    resolved = Path(filepath).resolve()
    stat_result = resolved.stat()
    key = _cache_key(resolved, stat_result, axis_order)
    directory = Path(memmap_directory)
    directory.mkdir(parents=True, exist_ok=True)
    memmap_path, sidecar_path = _cache_file_paths(directory, key)

    if memmap_path.exists() and sidecar_path.exists():
        cached = _read_cache_entry(memmap_path, sidecar_path)
        if cached is not None:
            logger.info(
                "Reusing the raw volume a previous run already decompressed: %s",
                memmap_path,
            )
            return cached
        logger.warning(
            "Ignoring an unreadable cached raw volume at %s; reloading from %s.",
            memmap_path,
            filepath,
        )

    image, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status = (
        load_image_with_voxel_size_2d_aware(
            filepath,
            input_format=input_format,
            axis_order=axis_order,
            h5_dataset_name=h5_dataset_name,
            use_memmap=True,
            memmap_path=memmap_path,
        )
    )
    _write_sidecar(
        sidecar_path, image, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status
    )
    return image, voxel_size_x, voxel_size_y, voxel_size_z, voxel_meta_status
