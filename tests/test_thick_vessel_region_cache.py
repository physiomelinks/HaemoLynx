"""The thick-vessel region kept on disk between runs
(``preprocessing/thick_vessel_region_cache.py``): keyed on everything that
decides it, read back exactly, and a miss on anything unreadable."""
from __future__ import annotations

import json

import numpy as np
import pytest

from haemolynx.preprocessing import ThickVesselRegionCache, thick_vessel_region_key
from haemolynx.preprocessing import thick_vessel_region_cache as module


def _mask() -> np.ndarray:
    z, y, _x = np.indices((6, 9, 12))
    return (z - 3) ** 2 + (y - 4) ** 2 <= 4


ARGUMENTS = dict(
    min_radius_um=3.0, voxel_size_zyx=(1.0, 0.5, 0.5), fill_mask_holes=True, wall_absorption_um=None
)


def test_the_key_reads_the_mask_and_every_argument_that_decides_the_region():
    mask = _mask()
    key = thick_vessel_region_key(mask, **ARGUMENTS)
    assert key == thick_vessel_region_key(mask.copy(), **ARGUMENTS)

    flipped = mask.copy()
    flipped[0, 0, 0] = True
    assert thick_vessel_region_key(flipped, **ARGUMENTS) != key
    for change in (
        {"min_radius_um": 2.0},
        {"voxel_size_zyx": (1.0, 0.5, 0.6)},
        {"fill_mask_holes": False},
        {"wall_absorption_um": 1.5},
    ):
        assert thick_vessel_region_key(mask, **{**ARGUMENTS, **change}) != key

    restriction = np.zeros(mask.shape, dtype=bool)
    restriction[2] = True
    restricted = thick_vessel_region_key(mask, **ARGUMENTS, restrict_thick_to_mask=restriction)
    assert restricted != key
    other = restriction.copy()
    other[3] = True
    assert thick_vessel_region_key(mask, **ARGUMENTS, restrict_thick_to_mask=other) != restricted
    # An all-False restriction is not "none": it is a restriction that finds nothing.
    nothing = np.zeros(mask.shape, dtype=bool)
    assert thick_vessel_region_key(mask, **ARGUMENTS, restrict_thick_to_mask=nothing) != key


def test_the_key_is_the_same_for_a_disk_backed_mask_and_one_of_another_dtype(tmp_path):
    mask = _mask()
    key = thick_vessel_region_key(mask, **ARGUMENTS)
    on_disk = np.memmap(tmp_path / "m.memmap", dtype=np.uint8, mode="w+", shape=mask.shape)
    on_disk[:] = mask
    assert thick_vessel_region_key(on_disk, **ARGUMENTS) == key
    assert thick_vessel_region_key(mask.astype(np.uint8) * 255, **ARGUMENTS) == key


def test_a_change_past_the_first_slab_changes_the_key():
    mask = np.zeros((module._DIGEST_SLAB + 8, 4, 4), dtype=bool)
    key = thick_vessel_region_key(mask, **ARGUMENTS)
    mask[-1, 1, 1] = True
    assert thick_vessel_region_key(mask, **ARGUMENTS) != key


def test_a_region_is_read_back_exactly_and_none_is_remembered_too(tmp_path):
    cache = ThickVesselRegionCache(tmp_path / "run.v2_thick_vessel_region")
    assert cache.lookup("k") == (False, None)

    region = _mask()
    cache.store("k", region, region.shape)
    # A dotted stem keeps its dot: the files sit beside the run's skeleton.
    assert cache.sidecar == tmp_path / "run.v2_thick_vessel_region.json"
    assert cache.packed == tmp_path / "run.v2_thick_vessel_region.npy"
    found, back = cache.lookup("k")
    assert found and back.dtype == bool and np.array_equal(back, region)
    assert cache.packed.stat().st_size < region.size  # one bit a voxel, not a byte
    assert cache.lookup("other") == (False, None)

    cache.store("k2", None, region.shape)
    assert cache.lookup("k2") == (True, None)
    assert not cache.packed.exists()
    assert cache.lookup("k") == (False, None)


def test_anything_unreadable_is_a_miss(tmp_path):
    region = _mask()
    cache = ThickVesselRegionCache(tmp_path / "r")
    cache.store("k", region, region.shape)
    cache.packed.write_bytes(cache.packed.read_bytes()[:-8])  # truncated
    assert cache.lookup("k") == (False, None)

    cache.store("k", region, region.shape)
    meta = json.loads(cache.sidecar.read_text())
    meta["format"] = module._CACHE_FORMAT + 1  # written by another version
    cache.sidecar.write_text(json.dumps(meta))
    assert cache.lookup("k") == (False, None)
    cache.sidecar.write_text("{not json")
    assert cache.lookup("k") == (False, None)
    cache.sidecar.unlink()
    assert cache.lookup("k") == (False, None)


def test_a_cache_that_cannot_be_written_only_warns(tmp_path, caplog):
    (tmp_path / "a_file").write_text("")
    cache = ThickVesselRegionCache(tmp_path / "a_file" / "r")  # no directory can be made there
    with caplog.at_level("WARNING"):
        cache.store("k", _mask(), None)
    assert "Could not keep the thick-vessel region" in caplog.text
    assert cache.lookup("k") == (False, None)


@pytest.mark.parametrize("shape, block", [((6, 9, 12), 50), ((13, 7, 5), 37), ((5, 3, 3), 9), ((4, 8, 8), 10**6)])
def test_low_ram_read_back_is_the_region_on_disk_a_slab_at_a_time(tmp_path, monkeypatch, shape, block):
    """Planes whose bits do not start on a byte, slabs of one plane or many:
    the same region, on disk."""
    from haemolynx.preprocessing import memmap_support

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", block)
    region = np.random.default_rng(sum(shape)).random(shape) > 0.6
    cache = ThickVesselRegionCache(tmp_path / "run_thick_vessel_region")
    cache.store("k", region, region.shape)

    found, in_ram = cache.lookup("k")
    found_low, on_disk = cache.lookup("k", use_memmap=True, memmap_directory=tmp_path)

    assert found and found_low
    assert isinstance(on_disk, np.memmap) and on_disk.dtype == bool
    assert np.array_equal(on_disk, in_ram) and np.array_equal(on_disk, region)
    memmap_support.release_memmap_array(on_disk)
    # The bits were read in place: storing again over the file still works.
    cache.store("k", ~region, region.shape)
    assert np.array_equal(cache.lookup("k")[1], ~region)


def test_low_ram_read_back_holds_no_volume_in_ram(tmp_path, monkeypatch):
    import tracemalloc

    from haemolynx.preprocessing import memmap_support

    monkeypatch.setattr(memmap_support, "LOW_MEMORY_BLOCK_VOXELS", 200_000)
    shape = (120, 160, 170)  # 3.3 M voxels
    region = np.zeros(shape, dtype=bool)
    region[10:100, 20:140, 30:150] = True
    cache = ThickVesselRegionCache(tmp_path / "run_thick_vessel_region")
    cache.store("k", region, shape)
    volume = region.size

    def peak(**kwargs):
        tracemalloc.start()
        try:
            found, back = cache.lookup("k", **kwargs)
            traced = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert found and np.array_equal(back, region)
        if isinstance(back, np.memmap):
            memmap_support.release_memmap_array(back)
        return traced

    assert peak() > 1.5 * volume, "in RAM: the bytes and the booleans"
    assert peak(use_memmap=True, memmap_directory=tmp_path) < 0.4 * volume


def test_discarding_the_entry_removes_both_files(tmp_path):
    cache = ThickVesselRegionCache(tmp_path / "E14_thick_vessel_region")
    mask = _mask()
    cache.store("key", mask, mask.shape)
    assert cache.sidecar.is_file() and cache.packed.is_file()
    assert set(cache.discard()) == {cache.sidecar, cache.packed}
    assert not cache.sidecar.exists() and not cache.packed.exists()
    assert cache.lookup("key") == (False, None)
    assert cache.discard() == ()
