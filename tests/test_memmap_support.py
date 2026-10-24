"""Disk-backed arrays behave like normal arrays and clean up after themselves.

No large data needed: these tests exercise the mechanism (allocate, write,
read back, release) on small shapes, not the memory savings themselves.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from haemolynx.preprocessing.memmap_support import (
    delete_dropped_memmap_files,
    new_memmap_array,
    release_memmap_array,
    temporary_memmap_array,
)


def test_a_new_memmap_array_has_the_requested_shape_and_dtype():
    array = new_memmap_array((3, 4, 5), np.int32)
    try:
        assert isinstance(array, np.memmap)
        assert array.shape == (3, 4, 5)
        assert array.dtype == np.int32
        assert Path(array.filename).is_file()
    finally:
        release_memmap_array(array)


def test_writing_to_a_memmap_array_round_trips_like_a_normal_array():
    array = new_memmap_array((2, 3), np.uint8)
    try:
        array[:] = 7
        array[0, 0] = 200
        assert array[0, 0] == 200
        assert np.array_equal(array[1:], np.full((1, 3), 7, dtype=np.uint8))
    finally:
        release_memmap_array(array)


def test_a_custom_directory_is_used_for_the_backing_file(tmp_path):
    array = new_memmap_array((2, 2), np.float64, directory=tmp_path)
    try:
        assert Path(array.filename).parent == tmp_path
    finally:
        release_memmap_array(array)


def test_a_custom_directory_is_created_if_it_does_not_already_exist(tmp_path):
    missing = tmp_path / "nested" / "memmap_dir"
    assert not missing.exists()

    array = new_memmap_array((2, 2), np.float64, directory=missing)
    try:
        assert missing.is_dir()
        assert Path(array.filename).parent == missing
    finally:
        release_memmap_array(array)


def test_release_memmap_array_deletes_the_backing_file_and_returns_its_path():
    array = new_memmap_array((2, 2), np.uint8)
    path = Path(array.filename)
    assert path.is_file()

    returned = release_memmap_array(array)

    assert returned == path
    assert not path.exists()


def test_temporary_memmap_array_is_usable_inside_the_block_and_cleaned_up_after():
    captured_path = None
    with temporary_memmap_array((4, 4), np.float32) as array:
        captured_path = Path(array.filename)
        array[:] = 1.5
        assert np.array_equal(array, np.full((4, 4), 1.5, dtype=np.float32))
        assert captured_path.is_file()

    assert not captured_path.exists()


def test_temporary_memmap_array_still_cleans_up_when_the_block_raises():
    captured_path = None

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with temporary_memmap_array((3, 3), np.int16) as array:
            captured_path = Path(array.filename)
            raise Boom("something went wrong inside the block")

    assert captured_path is not None
    assert not captured_path.exists()


# --- a memmap nobody releases (2026-10-18) -------------------------------------


def test_a_memmap_nobody_releases_has_its_file_deleted_once_it_and_its_views_are_gone(tmp_path):
    """The skeleton ``skeletonise`` hands on, the raw image the Diameters
    stage reads, and many a volume a test compares, were never released:
    every run left them in the temp folder. Windows will not delete a file
    while it is mapped, so there the file goes at the next memmap made (or
    as the process exits, ``_delete_dropped_files``)."""
    import gc

    from haemolynx.preprocessing import memmap_support

    array = new_memmap_array((4, 5), np.uint8, directory=tmp_path)
    view = array[1:]
    path = Path(array.filename)
    del array
    gc.collect()
    assert path.exists(), "a view of it still maps the file"

    del view
    gc.collect()
    fresh = new_memmap_array((2,), np.uint8, directory=tmp_path)

    assert not path.exists()
    assert str(path) not in memmap_support._DROPPED_FILES
    assert [p.name for p in tmp_path.iterdir()] == [Path(fresh.filename).name]
    release_memmap_array(fresh)


def test_a_released_memmap_leaves_nothing_for_the_clean_up_at_exit(tmp_path):
    import gc
    import os

    from haemolynx.preprocessing import memmap_support

    array = new_memmap_array((3, 3), np.uint8, directory=tmp_path)
    path = os.path.abspath(array.filename)
    assert path in memmap_support._FINALISERS

    release_memmap_array(array)
    del array
    gc.collect()

    assert path not in memmap_support._FINALISERS
    assert path not in memmap_support._DROPPED_FILES
    assert not os.path.exists(path)


def test_release_resident_pages_writes_every_volume_out_and_hands_its_pages_back(tmp_path):
    """Every live new_memmap_array volume and any memmap given -- a view of
    one included -- is written to its file, and on Windows taken out of this
    process's working set (flushing alone leaves it there, not counted free);
    the data reads back as written."""
    import sys

    import psutil

    from haemolynx.preprocessing.memmap_support import release_resident_pages

    made = new_memmap_array((64, 1024, 1024), np.uint8, directory=tmp_path)
    npy = np.lib.format.open_memmap(str(tmp_path / "given.npy"), mode="w+", dtype=np.uint8, shape=(32, 1024, 1024))
    made[...] = 3
    npy[...] = 5
    before = psutil.Process().memory_info().wset if sys.platform == "win32" else None

    release_resident_pages(npy[4:9], np.zeros(3), [1, 2])

    assert np.all(np.fromfile(made.filename, dtype=np.uint8) == 3), "written out"
    assert np.all(np.asarray(np.load(tmp_path / "given.npy", mmap_mode="r")) == 5)
    if before is not None:
        dropped = before - psutil.Process().memory_info().wset
        assert dropped > 0.8 * (made.nbytes + npy.nbytes), dropped
    assert np.all(made == 3) and np.all(npy == 5)
    release_memmap_array(made)
    del npy


def test_release_resident_pages_passes_over_a_released_volume(tmp_path):
    from haemolynx.preprocessing.memmap_support import release_resident_pages

    gone = new_memmap_array((4, 4), np.uint8, directory=tmp_path)
    release_memmap_array(gone)
    release_resident_pages(gone)


def test_dropped_scratch_files_are_deleted_on_asking(tmp_path):
    """Between a batch's images: every scratch volume nothing holds is gone,
    on Windows too, where its file waited while it was mapped."""
    array = new_memmap_array((4, 5), np.uint8, directory=tmp_path)
    array[:] = 1
    path = Path(array.filename)
    view = array[1:]
    del array
    assert path.exists(), "a view still holds the map"
    del view
    assert delete_dropped_memmap_files() == 0
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
