"""The thick-vessel region a run found, kept on disk for the next run.

Finding the region (:func:`~haemolynx.preprocessing.thick_vessels.thick_vessel_region`:
the hole fill, the inscribed-radius map, the geodesic body and its wall) is
the slow part of thickness-gated skeletonisation -- about four minutes on
E14.5 -- and every run found it again, from the same mask with the same
settings. The region depends on nothing but the mask's voxels and the
arguments that function takes, so an entry is keyed on a digest of exactly
those (:func:`thick_vessel_region_key`): a changed mask, voxel size, radius,
hole fill, wall absorption or restriction mask misses, and nothing else can
go stale. A mask with no thick region is remembered too, since finding
none costs as much as finding one.

One entry per run output, beside the saved skeleton, overwritten when its
key changes: ``{base}.json`` (the key, the shape, whether there is a region)
and ``{base}.npy`` (the region, one bit a voxel). A cache is an optimisation,
never a correctness requirement, so anything unreadable is a miss.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Written into every sidecar; an entry with any other is a miss. Raise it
#: whenever the region would be found differently from the same inputs.
_CACHE_FORMAT = 1

#: Slices digested at a time, so a disk-backed mask is never read whole.
_DIGEST_SLAB = 32


def _digest_mask(digest, mask: np.ndarray) -> None:
    digest.update(repr(tuple(int(n) for n in mask.shape)).encode())
    for start in range(0, mask.shape[0], _DIGEST_SLAB):
        slab = np.asarray(mask[start : start + _DIGEST_SLAB], dtype=bool)
        digest.update(np.packbits(slab).tobytes())


def thick_vessel_region_key(
    binary: np.ndarray,
    *,
    min_radius_um: float,
    voxel_size_zyx,
    fill_mask_holes: bool,
    wall_absorption_um: float | None,
    restrict_thick_to_mask: np.ndarray | None = None,
) -> str:
    """A digest of everything :func:`thick_vessel_region` reads: the mask's
    voxels and the arguments that decide the region. The low-RAM options
    (``use_memmap``, ``memmap_directory``) change where it is computed, not
    what, so they are not part of it."""
    digest = hashlib.sha1()
    digest.update(
        json.dumps(
            {
                "format": _CACHE_FORMAT,
                "min_radius_um": float(min_radius_um),
                "voxel_size_zyx": [float(v) for v in voxel_size_zyx],
                "fill_mask_holes": bool(fill_mask_holes),
                "wall_absorption_um": (
                    None if wall_absorption_um is None else float(wall_absorption_um)
                ),
                "restricted": restrict_thick_to_mask is not None,
            },
            sort_keys=True,
        ).encode()
    )
    _digest_mask(digest, binary)
    if restrict_thick_to_mask is not None:
        _digest_mask(digest, restrict_thick_to_mask)
    return digest.hexdigest()


class ThickVesselRegionCache:
    """The entry at *base* (a path without a suffix: ``{base}.json`` and
    ``{base}.npy``), looked up and stored by key."""

    def __init__(self, base: Path | str):
        self.base = Path(base)
        self.sidecar = Path(f"{self.base}.json")
        self.packed = Path(f"{self.base}.npy")

    def lookup(self, key: str) -> tuple[bool, np.ndarray | None]:
        """``(True, region)`` for the entry written under *key* -- the region
        ``None`` when the run that wrote it found no thick region --
        else ``(False, None)``."""
        try:
            meta = json.loads(self.sidecar.read_text())
            if meta.get("format") != _CACHE_FORMAT or meta.get("key") != key:
                return False, None
            shape = tuple(int(n) for n in meta["shape"])
            if not meta["region"]:
                return True, None
            packed = np.load(self.packed)
            count = int(np.prod(shape, dtype=np.int64))
            if packed.ndim != 1 or packed.dtype != np.uint8 or packed.size != (count + 7) // 8:
                return False, None
            region = np.unpackbits(packed, count=count).reshape(shape).astype(bool)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False, None
        logger.info("Reusing the thick-vessel region a previous run found: %s", self.packed)
        return True, region

    def store(self, key: str, region: np.ndarray | None, shape) -> None:
        """Write *region* (``None`` for no thick region in a mask of *shape*)
        under *key*, replacing whatever entry was there."""
        try:
            shape = tuple(int(n) for n in (shape if region is None else region.shape))
            self.base.parent.mkdir(parents=True, exist_ok=True)
            if region is None:
                if self.packed.exists():
                    self.packed.unlink()
            else:
                np.save(self.packed, np.packbits(np.asarray(region, dtype=bool)))
            self.sidecar.write_text(
                json.dumps(
                    {
                        "format": _CACHE_FORMAT,
                        "key": key,
                        "shape": list(shape),
                        "region": region is not None,
                    }
                )
            )
        except OSError as error:
            logger.warning("Could not keep the thick-vessel region at %s: %s", self.base, error)
