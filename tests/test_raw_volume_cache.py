"""``load_raw_volume_reusing_cache`` skips re-reading the source file on a
later run, but only when nothing that would change the result has changed.

The invalidation check itself needs to stat the *source* path on every call
(that is what lets a changed file be detected at all), so the source cannot
simply be deleted afterwards to prove reuse happened. Instead, every "reuses
the cache" test checks which file the returned array is actually backed by:
a hit is backed by the deterministic cache file (``_CACHE_PREFIX``-named,
under the cache directory), never by a freshly-read copy of the source.

A source modified in the last few seconds is never cached (see the module
docstring), so every test that expects an entry backdates its source first.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

from haemolynx.io.raw_volume_cache import (
    _CACHE_FORMAT,
    _CACHE_PREFIX,
    _RACY_WINDOW_NS,
    _cache_file_paths,
    _cache_key,
    discard_cached_raw_volume,
    load_raw_volume_reusing_cache,
)


def _backdate(path, seconds=60):
    """Stand in for a source written long before the run that loads it."""
    mtime_ns = time.time_ns() - int(seconds * 1_000_000_000)
    os.utime(path, ns=(mtime_ns, mtime_ns))


def _write_tiff(path, shape=(6, 10, 12), seed=0, backdate=True, **kwargs):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 255, size=shape, dtype=np.uint8)
    tifffile.imwrite(path, raw, **kwargs)
    if backdate:
        _backdate(path)
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
    new_raw = _write_tiff(source, seed=2, backdate=False)  # different content, same path

    image, *_ = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )

    assert np.array_equal(image, new_raw)


def test_a_rewrite_that_keeps_mtime_and_size_returns_the_new_volume(tmp_path):
    """Regression (2026-10-15): a same-size rewrite that lands in the same
    file-clock tick as the first write keeps its mtime and size, and the old
    (path, mtime, size) key served the old volume -- the rare flaky failure of
    the test above. Two TIFF writes with nothing between them shared both 92
    times in 200 on NTFS, but a load in between usually crosses a tick, so
    here both writes are given one identical, recent mtime to make the
    collision certain. No waiting; the old key failed this every time."""
    source = tmp_path / "volume.tif"
    cache_dir = tmp_path / "cache"
    mtime_ns = time.time_ns()

    _write_tiff(source, seed=1, backdate=False)
    os.utime(source, ns=(mtime_ns, mtime_ns))
    load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )
    new_raw = _write_tiff(source, seed=2, backdate=False)
    os.utime(source, ns=(mtime_ns, mtime_ns))

    image, *_ = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )

    assert np.array_equal(image, new_raw)


def test_a_recently_modified_source_is_loaded_but_not_cached(tmp_path):
    source = tmp_path / "volume.tif"
    raw = _write_tiff(source, backdate=False)
    cache_dir = tmp_path / "cache"

    image, *_ = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )

    assert isinstance(image, np.memmap)  # still kept off plain RAM
    assert np.array_equal(image, raw)
    assert not list(cache_dir.glob(f"{_CACHE_PREFIX}*"))


def test_a_source_older_than_the_racy_window_is_cached(tmp_path):
    source = tmp_path / "volume.tif"
    _write_tiff(source, backdate=False)
    _backdate(source, seconds=_RACY_WINDOW_NS / 1e9 + 1)
    cache_dir = tmp_path / "cache"

    image, *_ = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )

    memmap_path, _sidecar_path = _cache_files_for(cache_dir, source)
    assert Path(image.filename) == memmap_path


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

    # Transposed into a fresh memmap of its own (never the cache entry, whose
    # bytes are in the file's native order), so a non-canonical order stays
    # off plain RAM too.
    assert isinstance(image, np.memmap)
    cache_entry, _sidecar = _cache_files_for(cache_dir, source)
    assert Path(image.filename).resolve() != cache_entry.resolve()
    assert np.array_equal(image, np.transpose(raw, (2, 1, 0)))


def test_an_auto_axis_order_is_keyed_by_the_order_it_resolves_to(tmp_path):
    """Under image_axis_order="auto" a file with no axis labels is read zyx,
    so it shares the zyx entry (and a batch's clean-up finds it); one whose
    labels say x, y, z is transposed fresh, like an explicit "xyz"."""
    cache_dir = tmp_path / "cache"
    plain = tmp_path / "plain.tif"
    raw = _write_tiff(plain)
    load_raw_volume_reusing_cache(
        str(plain), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )
    image, *_ = load_raw_volume_reusing_cache(
        str(plain), input_format="tif", axis_order="auto", memmap_directory=cache_dir
    )
    entry = _cache_files_for(cache_dir, plain)
    assert Path(image.filename).resolve() == entry[0].resolve()
    assert np.array_equal(image, raw)

    labelled = tmp_path / "labelled.tif"
    stored = _write_tiff(labelled, shape=(12, 10, 6), metadata={"axes": "XYZ"})
    image, *_ = load_raw_volume_reusing_cache(
        str(labelled), input_format="tif", axis_order="auto", memmap_directory=cache_dir
    )
    assert np.array_equal(image, np.transpose(stored, (2, 1, 0)))
    assert not any(path.exists() for path in _cache_files_for(cache_dir, labelled))
    assert discard_cached_raw_volume(labelled, axis_order="auto", memmap_directory=cache_dir) == ()

    assert set(discard_cached_raw_volume(plain, axis_order="auto", memmap_directory=cache_dir)) == set(entry)


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
    _backdate(source)
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


def test_an_entry_from_an_older_version_is_a_miss_and_its_voxel_size_is_read_again(tmp_path):
    """Regression: an entry written before OME-TIFF voxel sizes were read from
    the OME header holds 1 micron on every axis, and reusing it would keep the
    run on that after the fix. Its sidecar has no format, so it misses."""
    source = tmp_path / "volume.ome.tif"
    raw = np.random.default_rng(0).integers(0, 255, size=(6, 10, 12), dtype=np.uint8)
    tifffile.imwrite(
        source, raw, ome=True,
        metadata={"axes": "ZYX", "PhysicalSizeX": 0.4, "PhysicalSizeY": 0.5, "PhysicalSizeZ": 2.0},
    )
    _backdate(source)
    cache_dir = tmp_path / "cache"

    load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )
    _memmap_path, sidecar_path = _cache_files_for(cache_dir, source)
    old = json.loads(sidecar_path.read_text())
    del old["format"]
    old.update(voxel_size_x=1.0, voxel_size_y=1.0, voxel_size_z=1.0)
    sidecar_path.write_text(json.dumps(old))

    image, vx, vy, vz, status = load_raw_volume_reusing_cache(
        str(source), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
    )

    assert (vx, vy, vz) == pytest.approx((0.4, 0.5, 2.0))
    assert status["status"] == "complete"
    assert np.array_equal(image, raw)
    assert json.loads(sidecar_path.read_text())["format"] == _CACHE_FORMAT


def test_compressed_tiff_can_be_loaded_and_reused(tmp_path):
    source = tmp_path / "compressed.tif"
    raw = np.zeros((5, 30, 30), dtype=np.uint8)
    raw[1:4, 5:20, 5:20] = 255
    tifffile.imwrite(source, raw, compression="zlib")
    _backdate(source)
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
    _backdate(source)
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
    _backdate(source)
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


def test_discarding_an_entry_removes_its_two_files_and_nothing_else(tmp_path):
    """A batch runs each image once: its decompressed copy is only disk taken
    once the image is done."""
    source, other = tmp_path / "volume.tif", tmp_path / "other.tif"
    _write_tiff(source, seed=1)
    _write_tiff(other, seed=2)
    cache_dir = tmp_path / "cache"
    for path in (source, other):
        image, *_rest = load_raw_volume_reusing_cache(
            str(path), input_format="tif", axis_order="zyx", memmap_directory=cache_dir
        )
        del image, _rest
    entry = _cache_files_for(cache_dir, source)

    removed = discard_cached_raw_volume(source, axis_order="zyx", memmap_directory=cache_dir)

    assert set(removed) == set(entry)
    assert not any(path.exists() for path in entry)
    assert all(path.exists() for path in _cache_files_for(cache_dir, other))
    assert discard_cached_raw_volume(source, axis_order="zyx", memmap_directory=cache_dir) == ()
    # Never cached: a non-canonical order, or a file that is not there.
    assert discard_cached_raw_volume(other, axis_order="xyz", memmap_directory=cache_dir) == ()
    assert discard_cached_raw_volume(tmp_path / "gone.tif", axis_order="zyx", memmap_directory=cache_dir) == ()
