"""Tests for io.load_and_validate_vessel_masks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ImageLynx.io import load_and_validate_vessel_masks


def _write_mask_tif(path: Path, shape: tuple[int, int, int]) -> None:
    tifffile.imwrite(path, np.zeros(shape, dtype=np.uint8))


def test_disabled_returns_none_without_paths() -> None:
    result = load_and_validate_vessel_masks(
        mask_role="large",
        enabled=False,
        use_ilastik=False,
        arteriole_mask_path=None,
        venule_mask_path=None,
        image_shape=(4, 5, 6),
        main_voxel_size_xyz=(1.0, 1.0, 1.0),
    )
    assert result == (None, None, None, None)


def test_ilastik_requires_enabled_flag() -> None:
    with pytest.raises(ValueError, match="use_ilastik_large_vessel_segmentation=True requires"):
        load_and_validate_vessel_masks(
            mask_role="large",
            enabled=False,
            use_ilastik=True,
            arteriole_mask_path=None,
            venule_mask_path=None,
            image_shape=(4, 5, 6),
            main_voxel_size_xyz=(1.0, 1.0, 1.0),
        )


def test_loads_and_validates_matching_masks(tmp_path: Path) -> None:
    shape = (4, 5, 6)
    arteriole_path = tmp_path / "arteriole.tif"
    venule_path = tmp_path / "venule.tif"
    _write_mask_tif(arteriole_path, shape)
    _write_mask_tif(venule_path, shape)

    arteriole_mask, venule_mask, art_vox, ven_vox = load_and_validate_vessel_masks(
        mask_role="large",
        enabled=True,
        use_ilastik=False,
        arteriole_mask_path=arteriole_path,
        venule_mask_path=venule_path,
        image_shape=shape,
        main_voxel_size_xyz=(1.0, 1.0, 1.0),
    )

    assert arteriole_mask is not None and venule_mask is not None
    assert arteriole_mask.shape == shape
    assert venule_mask.shape == shape
    assert art_vox == (1.0, 1.0, 1.0)
    assert ven_vox == (1.0, 1.0, 1.0)


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    arteriole_path = tmp_path / "arteriole.tif"
    venule_path = tmp_path / "venule.tif"
    _write_mask_tif(arteriole_path, (4, 5, 6))
    _write_mask_tif(venule_path, (4, 5, 6))

    with pytest.raises(ValueError, match="large_arteriole_mask shape does not match"):
        load_and_validate_vessel_masks(
            mask_role="large",
            enabled=True,
            use_ilastik=False,
            arteriole_mask_path=arteriole_path,
            venule_mask_path=venule_path,
            image_shape=(3, 5, 6),
            main_voxel_size_xyz=(1.0, 1.0, 1.0),
        )
