"""``load_raw_volume_reusing_cache`` skips re-reading the source file on a
later run, but only when nothing that would change the result has changed.

The invalidation check itself needs to stat the *source* path on every call
(that is what lets a changed file be detected at all), so the source cannot
simply be deleted afterwards to prove reuse happened. Instead, every "reuses
the cache" test checks which file the returned array is actually backed by:
a hit is backed by the deterministic cache file (``_CACHE_PREFIX``-named,
under the cache directory), never by a freshly-read copy of the source.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from haemolynx.io.raw_volume_cache import (
    _CACHE_PREFIX,
    _cache_file_paths,
    _cache_key,
    load_raw_volume_reusing_cache,
)


def _write_tiff(path, shape=(6, 10, 12), seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 255, size=shape, dtype=np.uint8)
    tifffile.imwrite(path, raw, **kwargs)
    return raw


def _cache_files_for(tmp_path, source_path, axis_order="zyx"):
    resolved = source_path.resolve()
    key = _cache_key(resolved, resolved.stat(), axis_order)
    return _cache_file_paths(tmp_path, key)


def test_a_cache_miss_loads_and_writes_a_reusable_entry(tmp_path):
    source = tmp_path / "volume.tif"
    raw = _write_tiff(source)
    cache_dir = tmp_path / "cache"

    image, vx, vy, vz, status = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )

    assert isinstance(image, np.memmap)
    assert np.array_equal(image, raw)
    memmap_path, sidecar_path = _cache_files_for(cache_dir, source)
    assert memmap_path.exists()
    assert sidecar_path.exists()
    assert memmap_path.name.startswith(_CACHE_PREFIX)


def test_a_second_load_is_backed_by_the_cache_file_not_a_fresh_read(tmp_path):
    source = tmp_path / "volume.tif"
    raw = _write_tiff(source)
    cache_dir = tmp_path / "cache"

    first, *_ = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )
    memmap_path, _sidecar_path = _cache_files_for(cache_dir, source)
    assert Path(first.filename) == memmap_path

    second, vx, vy, vz, status = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )

    assert isinstance(second, np.memmap)
    assert Path(second.filename) == memmap_path
    assert np.array_equal(second, raw)


def test_a_changed_source_file_misses_and_rewrites_the_cache(tmp_path):
    source = tmp_path / "volume.tif"
    _write_tiff(source, seed=1)
    cache_dir = tmp_path / "cache"

    load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )
    new_raw = _write_tiff(source, seed=2)  # different content, same path

    image, *_ = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )

    assert np.array_equal(image, new_raw)


def test_a_different_axis_order_does_not_reuse_the_zyx_cache_entry(tmp_path):
    source = tmp_path / "volume.tif"
    raw = _write_tiff(source, shape=(6, 10, 12))
    cache_dir = tmp_path / "cache"

    load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )
    # A non-canonical axis_order always misses (see module docstring) --
    # this must fall through to an ordinary load, not raise or reuse the
    # zyx entry under a mismatched shape.
    image, *_ = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="xyz", memmap_directory=cache_dir
    )

    assert not isinstance(image, np.memmap)  # apply_axis_order copies for non-canonical order
    assert np.array_equal(image, np.transpose(raw, (2, 1, 0)))


def test_a_truncated_cache_file_is_treated_as_a_miss_and_rebuilt(tmp_path):
    source = tmp_path / "volume.tif"
    raw = _write_tiff(source)
    cache_dir = tmp_path / "cache"

    load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )
    memmap_path, _sidecar_path = _cache_files_for(cache_dir, source)
    memmap_path.write_bytes(b"\x00")  # corrupt: wrong size for the stored shape/dtype

    image, *_ = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )

    assert np.array_equal(image, raw)


def test_a_missing_sidecar_is_treated_as_a_miss_and_rebuilt(tmp_path):
    source = tmp_path / "volume.tif"
    raw = _write_tiff(source)
    cache_dir = tmp_path / "cache"

    load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )
    _memmap_path, sidecar_path = _cache_files_for(cache_dir, source)
    sidecar_path.unlink()

    image, *_ = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )

    assert np.array_equal(image, raw)


def test_voxel_metadata_round_trips_through_the_cache(tmp_path):
    source = tmp_path / "volume.tif"
    raw = _write_tiff(source)
    tifffile.imwrite(
        source, raw, resolution=(4.0, 5.0), metadata={"spacing": 2.0, "unit": "um"}
    )
    cache_dir = tmp_path / "cache"

    _image, vx, vy, vz, status = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )
    image2, vx2, vy2, vz2, status2 = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )

    memmap_path, _sidecar_path = _cache_files_for(cache_dir, source)
    assert Path(image2.filename) == memmap_path  # came from the cache, not a fresh read
    assert (vx2, vy2, vz2) == (vx, vy, vz)
    assert status2 == status


def test_compressed_tiff_can_be_loaded_and_reused(tmp_path):
    source = tmp_path / "compressed.tif"
    raw = np.zeros((5, 30, 30), dtype=np.uint8)
    raw[1:4, 5:20, 5:20] = 255
    tifffile.imwrite(source, raw, compression="zlib")
    cache_dir = tmp_path / "cache"

    image, *_ = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )
    assert np.array_equal(image, raw)

    image2, *_ = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )
    memmap_path, _sidecar_path = _cache_files_for(cache_dir, source)
    assert Path(image2.filename) == memmap_path
    assert np.array_equal(image2, raw)


def test_h5_source_can_be_loaded_and_reused(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "volume.h5"
    raw = np.random.default_rng(0).integers(0, 255, size=(6, 10, 12), dtype=np.uint8)
    with h5py.File(source, "w") as handle:
        handle.create_dataset("data", data=raw)
    cache_dir = tmp_path / "cache"

    first, *_ = load_raw_volume_reusing_cache(
        str(source), input_format="h5", axis_order="zyx", memmap_directory=cache_dir
    )
    image, *_ = load_raw_volume_reusing_cache(
        str(source), input_format="h5", axis_order="zyx", memmap_directory=cache_dir
    )

    memmap_path, _sidecar_path = _cache_files_for(cache_dir, source)
    assert Path(first.filename) == memmap_path  # keeps `first` referenced too
    assert Path(image.filename) == memmap_path
    assert np.array_equal(image, raw)


def test_a_promoted_2d_tiff_round_trips_through_the_cache(tmp_path):
    source = tmp_path / "flat.tif"
    raw = np.random.default_rng(0).integers(0, 255, size=(10, 14), dtype=np.uint8)
    tifffile.imwrite(source, raw)
    cache_dir = tmp_path / "cache"

    image, *_ , status = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )
    assert image.shape == (1, 10, 14)
    assert status["promoted_from_2d"] is True

    image2, *_, status2 = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )
    memmap_path, _sidecar_path = _cache_files_for(cache_dir, source)
    assert Path(image2.filename) == memmap_path
    assert np.array_equal(image2[0], raw)
    assert status2["promoted_from_2d"] is True
