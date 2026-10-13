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
