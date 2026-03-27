"""Cache helpers for cleaned large/small vessel masks used by visualization."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

MASK_CACHE_VERSION = 1


def save_cleaned_mask_cache(
    cache_path: Path,
    *,
    image_shape_zyx: tuple[int, int, int],
    large_arteriole_mask: np.ndarray | None,
    large_venule_mask: np.ndarray | None,
    small_arteriole_mask: np.ndarray | None,
    small_venule_mask: np.ndarray | None,
) -> bool:
    """Persist cleaned mask volumes for later visualization reuse."""
    if (
        large_arteriole_mask is None
        and large_venule_mask is None
        and small_arteriole_mask is None
        and small_venule_mask is None
    ):
        return False
    payload = {
        "version": int(MASK_CACHE_VERSION),
        "image_shape_zyx": tuple(int(v) for v in image_shape_zyx),
        "large_arteriole_mask": (
            None if large_arteriole_mask is None else large_arteriole_mask.astype(bool, copy=True)
        ),
        "large_venule_mask": (
            None if large_venule_mask is None else large_venule_mask.astype(bool, copy=True)
        ),
        "small_arteriole_mask": (
            None if small_arteriole_mask is None else small_arteriole_mask.astype(bool, copy=True)
        ),
        "small_venule_mask": (
            None if small_venule_mask is None else small_venule_mask.astype(bool, copy=True)
        ),
    }
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return True


def load_cleaned_mask_cache(
    cache_path: Path,
    *,
    expected_image_shape_zyx: tuple[int, int, int],
) -> dict[str, Any] | None:
    """Load cleaned masks if cache file exists and metadata is compatible."""
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version", -1)) != int(MASK_CACHE_VERSION):
        return None
    cached_shape = tuple(int(v) for v in payload.get("image_shape_zyx", ()))
    if cached_shape != tuple(int(v) for v in expected_image_shape_zyx):
        return None
    return payload

