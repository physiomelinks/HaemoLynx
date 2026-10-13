"""What a raw image's channel stains: the plasma in the lumen, or the wall.

``raw_image_stain="auto"`` is answered here, from the raw channel and the
segmentation together. A luminal (plasma) label fills the channel blood flows
in, so across a segmented vessel it is brightest in the middle and falls off
past the mask's edge. An endothelial label (claudin-5, CD31, isolectin...)
images the wall: a bright ring round a dark lumen, so it is brightest just
outside a lumen mask and dim deep inside it.

Only vessels wide enough to have a middle are read -- across a capillary the
blur fills an endothelial ring in and it looks luminal -- in blocks round
randomly chosen mask voxels, so the cost does not grow with the volume. Each
block's distance transforms give the ``core`` (the middle of its widest
vessels: at least ``min_core_depth_um`` and ``core_fraction`` of the block's
deepest point inside the mask), the ``wall`` (the band within
``wall_reach_um`` either side of the mask's edge) and the ``background``
(further out). On E14.5 the claudin-5 channel peaks in that band at about
twelve times its level 5 um inside a vessel, while the dextran channel falls
from the middle out. The
answer is the sign of :attr:`StainReading.contrast`, ``(core - wall) /
(|core| + |wall|)`` above background, when its size clears
:data:`DECISIVE_CONTRAST` and the brighter of the two stands clear of the
background's own scatter; otherwise it says it cannot tell.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

PLASMA_COLUMN = "plasma_column"
ENDOTHELIAL = "endothelial"
#: The stains ``raw_image_stain`` names, besides ``auto``.
STAINS = (PLASMA_COLUMN, ENDOTHELIAL)
#: How far the contrast must lean either way to decide. With 64 blocks, on
#: E14.5 the dextran channel reads +0.25 to +0.33 and the claudin-5 one -0.16
#: to -0.44 over five seeds (about 5 s each); synthetic vessels with that
#: stack's blur read further out. Noise, or an image unrelated to the mask,
#: is stopped by SIGNAL_OVER_SPREAD.
DECISIVE_CONTRAST = 0.08
#: The margin round each block its distance transforms are taken over.
MARGIN_UM = 10.0
#: How far, in the background's own robust spread, the brighter of the core
#: and the wall must stand above it for the contrast to mean anything.
SIGNAL_OVER_SPREAD = 1.0


@dataclass(frozen=True)
class StainReading:
    """How a raw channel reads across the segmented vessels."""

    #: ``plasma_column``, ``endothelial``, or None when it cannot tell.
    stain: Optional[str]
    #: ``(core - wall) / (|core| + |wall|)``, each above the background:
    #: positive for a luminal label, negative for an endothelial one.
    contrast: float
    core: float
    wall: float
    background: float
    #: The background's robust spread (1.4826 x its median absolute deviation).
    background_spread: float
    blocks: int
    core_voxels: int
    wall_voxels: int

    def describe(self) -> str:
        what = {PLASMA_COLUMN: "a plasma (luminal) label", ENDOTHELIAL: "an endothelial (wall) label"}
        verdict = what.get(self.stain, "neither clearly")
        return (
            f"reads as {verdict}: contrast {self.contrast:+.2f} (deep in the vessels "
            f"{self.core:.4g}, just outside them {self.wall:.4g}, background {self.background:.4g}; "
            f"{self.core_voxels} and {self.wall_voxels} voxels in {self.blocks} blocks)"
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_raw_stain(
    raw: np.ndarray,
    mask: np.ndarray,
    voxel_size_zyx,
    *,
    min_core_depth_um: float = 3.0,
    core_fraction: float = 0.6,
    wall_reach_um: float = 1.0,
    background_from_um: float = 4.0,
    blocks: int = 64,
    block_um: float = 40.0,
    seed: int = 0,
) -> StainReading:
    """Read what *raw* stains, from how it lies across *mask*'s vessels (see
    the module docstring). *raw* and *mask* share one grid; either may be a
    memory map."""
    from scipy import ndimage

    if tuple(np.shape(raw)) != tuple(np.shape(mask)):
        # Read with the mask's own blocks on another grid -- a raw image not
        # cut to the run's crop box -- the levels come from the wrong place.
        raise ValueError(
            f"The raw image {tuple(np.shape(raw))} and the mask {tuple(np.shape(mask))} differ in "
            "shape: the stain is read where the mask's vessels are, so the two must share one grid"
        )
    vs = np.asarray(voxel_size_zyx, dtype=float)
    shape = np.asarray(mask.shape)
    half = np.maximum(np.ceil(block_um / 2.0 / vs).astype(int), 2)
    margin = np.ceil(MARGIN_UM / vs).astype(int)
    rng = np.random.default_rng(seed)
    # Candidate centres: mask voxels, read a slab at a time so a memory map is
    # never pulled in whole.
    candidates = _sample_mask_voxels(mask, rng, count=blocks * 40)
    core_values, wall_values, background_values = [], [], []
    used = 0
    for centre in candidates:
        if used >= blocks:
            break
        # Distances are taken over a margin round the block and read inside
        # it: within a bare crop, a vessel the crop cuts would look deeper
        # than it is, its wall being outside the crop.
        lo = np.maximum(centre - half - margin, 0)
        hi = np.minimum(centre + half + margin + 1, shape)
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        inner = np.zeros(tuple(hi - lo), dtype=bool)
        inner[tuple(slice(int(max(c - h - a, 0)), int(c + h + 1 - a)) for c, h, a in zip(centre, half, lo))] = True
        vessel = np.asarray(mask[box], dtype=bool)
        depth = ndimage.distance_transform_edt(vessel, sampling=vs)
        core = inner & (depth >= max(min_core_depth_um, core_fraction * float(depth[inner].max())))
        if core.sum() < 8:
            continue
        outside = ndimage.distance_transform_edt(~vessel, sampling=vs)
        wall = inner & ((vessel & (depth <= wall_reach_um)) | (~vessel & (outside <= wall_reach_um)))
        background = inner & (~vessel) & (outside >= background_from_um)
        if wall.sum() < 8 or background.sum() < 8:
            continue
        values = np.asarray(raw[box], dtype=float)
        core_values.append(values[core])
        wall_values.append(values[wall])
        background_values.append(values[background])
        used += 1
    if not used:
        return StainReading(None, 0.0, np.nan, np.nan, np.nan, np.nan, 0, 0, 0)
    core_all, wall_all, background_all = (np.concatenate(v) for v in (core_values, wall_values, background_values))
    background_level = float(np.median(background_all))
    spread = float(1.4826 * np.median(np.abs(background_all - background_level)))
    core_level, wall_level = float(np.median(core_all)), float(np.median(wall_all))
    lift_core, lift_wall = core_level - background_level, wall_level - background_level
    scale = abs(lift_core) + abs(lift_wall)
    contrast = float((lift_core - lift_wall) / scale) if scale > 0 else 0.0
    stain = None
    if max(abs(lift_core), abs(lift_wall)) < SIGNAL_OVER_SPREAD * spread:
        pass  # neither stands out of the background: nothing to read
    elif contrast >= DECISIVE_CONTRAST:
        stain = PLASMA_COLUMN
    elif contrast <= -DECISIVE_CONTRAST:
        stain = ENDOTHELIAL
    return StainReading(
        stain, contrast, core_level, wall_level, background_level, spread, used,
        int(core_all.size), int(wall_all.size),
    )


def _sample_mask_voxels(mask: np.ndarray, rng: np.random.Generator, *, count: int) -> list[np.ndarray]:
    """Up to *count* mask voxels, spread over the slabs that hold any, read a
    slab at a time so a memory map is never pulled in whole. A stack with
    few slabs -- one, for a 2D image -- gives each of them several, so the
    reading still has blocks to choose from; a deep one gives each a few."""
    slabs = [z for z in range(mask.shape[0]) if np.any(mask[z])]
    if not slabs:
        return []
    size = min(count, max(4 * len(slabs), count // 4))
    chosen = rng.choice(slabs, size=size, replace=size > len(slabs))
    picks = []
    for z in np.unique(chosen):
        yx = np.argwhere(np.asarray(mask[z], dtype=bool))
        for row in yx[rng.integers(len(yx), size=int(np.count_nonzero(chosen == z)))]:
            picks.append(np.asarray([z, row[0], row[1]]))
    rng.shuffle(picks)
    return picks[:count]


def resolve_stain(
    setting: str, raw: np.ndarray, mask: np.ndarray, voxel_size_zyx, **read_options
) -> tuple[str, Optional[StainReading]]:
    """``(stain, reading)`` for a ``raw_image_stain`` *setting*: the setting
    itself unless it is ``auto``; then read from *raw* and *mask*, with a
    reading that cannot tell taken as a plasma label -- the segmentation's
    own kind -- and the reading returned so a caller can say so."""
    if setting in STAINS:
        return setting, None
    reading = read_raw_stain(raw, mask, voxel_size_zyx, **read_options)
    return (reading.stain or PLASMA_COLUMN), reading
