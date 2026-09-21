"""Disk-backed arrays behave like normal arrays and clean up after themselves.

No large data needed: these tests exercise the mechanism (allocate, write,
read back, release) on small shapes, not the memory savings themselves.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from haemolynx.preprocessing.memmap_support import (
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
