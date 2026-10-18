"""Downsample on import: the segmented image read onto a coarser grid, and
every volume aligned with it reduced the same way.

Two choices on the Input tab, each ``none``, ``2x``, ``4x``, ``8x`` or
``16x`` (:data:`DOWNSAMPLE_CHOICES`):

* ``segmented_image_downsample`` (under ``segmentation_source``) -- the
  segmented image is reduced as it is loaded, after ``crop_on_import``'s box
  is cut, and every stage runs on the reduced image: the skeleton, the graph,
  the diameters and the exports are all on that grid, in the same microns.
  Every volume that has to lie on the segmentation voxel for voxel -- the
  large and small vessel masks, the pericyte mask, the cell mask -- is reduced
  by the same blocks (:func:`mask_volume_like_graph`), so each still lands on
  the vessels it belongs to.
* ``raw_image_downsample`` (under ``raw_image_stain``) -- how the raw image is
  read. Everything that reads it does so beside the segmentation and needs
  it on the segmentation's grid (the stain reading, the raw evidence, the
  diameters, the tissue volume, the checks), so a finer choice than the
  segmented image's is read at the segmented image's, and a coarser one is
  reduced by its own blocks and each coarse voxel spread over the grid
  (:func:`raw_volume_like_graph`). ilastik reads the whole file whatever it
  says: its project was trained on it.

A factor applies to the finest axes; an axis whose voxels are already
coarser is reduced by less, so the grid stays as near isotropic as the
factors allow (:func:`axis_downsample_factors`, the rule the settings
optimiser's own downsampling uses). A mask is reduced by block *max* -- a
coarse voxel is vessel if any voxel in it was, so a thin vessel survives,
though two vessels closer together than a coarse voxel merge -- and an
intensity image by block *mean*, in its own dtype. A block that runs past the
image's edge is reduced over the voxels it has.

Positions: a coarse voxel's index times the coarse voxel size is where the
first voxel of its block was. The network records what it was built on
(:data:`GRAPH_ATTRIBUTE`, :func:`record_on_graph`), and every later volume
read beside the image is placed by that record.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .axis_order import voxel_size_xyz_from_zyx, voxel_size_zyx_from_xyz
from .crop import crop_volume_like_graph

logger = logging.getLogger(__name__)

__all__ = [
    "DOWNSAMPLE_CHOICES",
    "DOWNSAMPLE_FACTORS",
    "GRAPH_ATTRIBUTE",
    "NONE",
    "RAW_SETTING",
    "SEGMENTED_SETTING",
    "Downsample",
    "axis_downsample_factors",
    "block_reduce_volume",
    "downsample_factor",
    "downsample_for",
    "downsample_from_settings",
    "downsample_raw_volume",
    "downsample_volume",
    "graph_downsample",
    "graph_raw_downsample_factor",
    "mask_volume_like_graph",
    "raw_volume_like_graph",
    "record_on_graph",
    "spread_onto_grid",
]

#: What the two Input tab choices offer, "none" (no reduction) first.
DOWNSAMPLE_CHOICES: tuple[str, ...] = ("none", "2x", "4x", "8x", "16x")
NONE = DOWNSAMPLE_CHOICES[0]
#: The factors behind them, 1 first.
DOWNSAMPLE_FACTORS: tuple[int, ...] = (1, 2, 4, 8, 16)

SEGMENTED_SETTING = "segmented_image_downsample"
RAW_SETTING = "raw_image_downsample"

#: The graph attribute holding what the network was built on
#: (:meth:`Downsample.to_record`, plus the raw image's own factor); absent
#: when neither image is reduced.
GRAPH_ATTRIBUTE = "downsample_on_import"

_CHOICE_TO_FACTOR = dict(zip(DOWNSAMPLE_CHOICES, DOWNSAMPLE_FACTORS))


def downsample_factor(choice: Any) -> int:
    """The factor a setting's value asks for: ``"4x"`` or ``4`` or ``"4"`` is
    4; ``"none"``, ``None`` and 1 are 1. Anything else raises ``ValueError``."""
    if choice is None:
        return 1
    if isinstance(choice, str):
        text = choice.strip().lower()
        if text in _CHOICE_TO_FACTOR:
            return _CHOICE_TO_FACTOR[text]
        if text.endswith("x"):
            text = text[:-1]
        try:
            choice = int(text)
        except ValueError:
            raise ValueError(
                f"A downsample choice is one of {', '.join(DOWNSAMPLE_CHOICES)}; got {choice!r}."
            ) from None
    if isinstance(choice, (bool, np.bool_)):
        raise ValueError(f"A downsample choice is one of {', '.join(DOWNSAMPLE_CHOICES)}; got {choice!r}.")
    factor = int(choice)
    if factor != choice or factor not in DOWNSAMPLE_FACTORS:
        raise ValueError(
            f"A downsample factor is one of {', '.join(str(f) for f in DOWNSAMPLE_FACTORS)}; got {choice!r}."
        )
    return factor


def axis_downsample_factors(
    factor: int, voxel_size_zyx: Optional[Sequence[float]] = None
) -> tuple[int, ...]:
    """How much each axis is reduced by for a *factor*.

    *factor* applies to the finest axes; a coarser one is reduced by only as
    much as brings its voxels nearest the same size (at least 1), so the
    grid is as close to isotropic as the factors allow. On a
    0.5 x 0.5 x 2 um stack, factor 4 gives 2 um voxels on every axis, where
    reducing each axis by 4 gave 2 x 2 x 8 um -- a vessel's z extent four
    times coarser than the resolution it was judged at in-plane. Without
    *voxel_size_zyx*, every axis by *factor*.
    """
    factor = max(1, int(factor))
    if voxel_size_zyx is None:
        return (factor, factor, factor)
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    finest = float(spacing.min())
    return tuple(max(1, int(round(factor * finest / float(s)))) for s in spacing)


def _reduce_axis(a: np.ndarray, block: int, axis: int, reduce: str) -> np.ndarray:
    """*a* reduced along *axis* by *block* voxels at a time, the last block
    over whatever voxels remain: block max, or block mean in float64."""
    if block <= 1:
        return a
    moved = np.moveaxis(a, axis, 0)
    n = moved.shape[0]
    full = (n // block) * block
    parts = []
    if full:
        head = moved[:full].reshape((full // block, block) + moved.shape[1:])
        parts.append(head.max(axis=1) if reduce == "max" else head.mean(axis=1, dtype=np.float64))
    if full < n:
        tail = moved[full:]
        parts.append(
            tail.max(axis=0, keepdims=True)
            if reduce == "max"
            else tail.mean(axis=0, keepdims=True, dtype=np.float64)
        )
    reduced = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
    return np.moveaxis(reduced, 0, axis)


def _cast(plane: np.ndarray, dtype: np.dtype, reduce: str) -> np.ndarray:
    """A reduced plane back in the volume's dtype: a mean of integers rounded
    to the nearest, and held within the type's range."""
    if plane.dtype == dtype:
        return plane
    if reduce == "mean" and np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(plane), info.min, info.max).astype(dtype)
    if reduce == "mean" and dtype == bool:
        return plane >= 0.5
    return plane.astype(dtype)


def block_reduce_volume(
    volume: np.ndarray,
    blocks: Sequence[int],
    reduce: str = "max",
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """*volume* with every block of *blocks* voxels (one per axis) become one:
    its max (``reduce="max"``, for a mask: foreground if any voxel was) or its
    mean (``"mean"``, for an intensity image, rounded back into its dtype).

    The result has ``ceil(n / block)`` voxels along each axis -- a block running
    past the edge is reduced over the voxels it has, never padded. *volume*
    itself comes back when every block is 1. Reduced a slab of blocks at a
    time along axis 0, so a disk-backed volume is read once in order; with
    *use_memmap* the result is a new disk-backed array.
    """
    if reduce not in ("max", "mean"):
        raise ValueError(f"reduce must be 'max' or 'mean'; got {reduce!r}.")
    blocks = tuple(max(1, int(b)) for b in blocks)
    if len(blocks) != volume.ndim:
        raise ValueError(f"{len(blocks)} block sizes for a {volume.ndim}-D volume.")
    if max(blocks) <= 1:
        return volume
    out_shape = tuple(-(-int(n) // b) for n, b in zip(volume.shape, blocks))
    dtype = volume.dtype
    if use_memmap:
        from haemolynx.preprocessing.memmap_support import new_memmap_array

        out = new_memmap_array(out_shape, dtype, directory=memmap_directory)
    else:
        out = np.empty(out_shape, dtype=dtype)
    first = blocks[0]
    for k in range(out_shape[0]):
        slab = np.asarray(volume[k * first:(k + 1) * first])
        plane = _reduce_axis(slab, first, 0, reduce)
        for axis in range(1, volume.ndim):
            plane = _reduce_axis(plane, blocks[axis], axis, reduce)
        out[k] = _cast(plane[0], dtype, reduce)
    return out


@dataclass(frozen=True)
class Downsample:
    """How a volume of *source_shape_zyx* voxels, *source_voxel_size_zyx*
    microns each, is reduced: *axis_factors_zyx* voxels become one along each
    axis (:func:`axis_downsample_factors` of *factor*)."""

    factor: int
    axis_factors_zyx: tuple[int, int, int]
    source_shape_zyx: tuple[int, int, int]
    source_voxel_size_zyx: tuple[float, float, float]

    @property
    def is_identity(self) -> bool:
        return max(self.axis_factors_zyx) <= 1

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(-(-n // b) for n, b in zip(self.source_shape_zyx, self.axis_factors_zyx))  # type: ignore[return-value]

    @property
    def voxel_size_zyx(self) -> tuple[float, float, float]:
        """The reduced grid's voxel size."""
        return self.scale_voxel_size_zyx(self.source_voxel_size_zyx)

    @property
    def source_voxel_size_xyz(self) -> tuple[float, float, float]:
        return voxel_size_xyz_from_zyx(self.source_voxel_size_zyx)

    def scale_voxel_size_zyx(self, voxel_size_zyx: Sequence[float]) -> tuple[float, float, float]:
        """A ``(z, y, x)`` voxel size on the source grid, on the reduced one."""
        return tuple(float(v) * b for v, b in zip(voxel_size_zyx, self.axis_factors_zyx))  # type: ignore[return-value]

    def scale_voxel_size_xyz(self, voxel_size_xyz: Sequence[float]) -> tuple[float, float, float]:
        """An ``(x, y, z)`` voxel size on the source grid, on the reduced one."""
        return voxel_size_xyz_from_zyx(
            self.scale_voxel_size_zyx(voxel_size_zyx_from_xyz(tuple(float(v) for v in voxel_size_xyz)))
        )

    def to_record(self) -> dict[str, Any]:
        """What a graph keeps (:data:`GRAPH_ATTRIBUTE`): plain tuples, so it
        pickles and saves with a run."""
        return {
            "factor": int(self.factor),
            "axis_factors_zyx": tuple(int(b) for b in self.axis_factors_zyx),
            "source_shape_zyx": tuple(int(n) for n in self.source_shape_zyx),
            "shape_zyx": tuple(int(n) for n in self.shape_zyx),
            "source_voxel_size_zyx": tuple(float(v) for v in self.source_voxel_size_zyx),
            "voxel_size_zyx": tuple(float(v) for v in self.voxel_size_zyx),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Downsample":
        return cls(
            int(record["factor"]),
            tuple(int(b) for b in record["axis_factors_zyx"]),  # type: ignore[arg-type]
            tuple(int(n) for n in record["source_shape_zyx"]),  # type: ignore[arg-type]
            tuple(float(v) for v in record["source_voxel_size_zyx"]),  # type: ignore[arg-type]
        )

    def describe(self) -> str:
        """The reduction as a user reads it."""
        blocks = " x ".join(str(b) for b in reversed(self.axis_factors_zyx))
        before = " x ".join(str(n) for n in reversed(self.source_shape_zyx))
        after = " x ".join(str(n) for n in reversed(self.shape_zyx))
        return (
            f"{self.factor}x: every {blocks} voxels (x, y, z) become one; "
            f"{before} -> {after} voxels"
        )


def downsample_for(
    factor: int, shape_zyx: Sequence[int], voxel_size_zyx: Sequence[float]
) -> Downsample:
    """The :class:`Downsample` *factor* makes of a volume of *shape_zyx*
    voxels, *voxel_size_zyx* microns each (an identity at factor 1)."""
    shape = tuple(int(n) for n in shape_zyx)
    if len(shape) != 3:
        raise ValueError(f"Downsample on import needs a 3D volume; got shape {shape}.")
    spacing = tuple(float(v) for v in voxel_size_zyx)
    return Downsample(int(factor), axis_downsample_factors(factor, spacing), shape, spacing)  # type: ignore[arg-type]


def downsample_from_settings(
    settings: Mapping[str, Any],
    shape_zyx: Sequence[int],
    voxel_size_zyx: Sequence[float],
    *,
    name: str = SEGMENTED_SETTING,
) -> Downsample | None:
    """The reduction the setting *name* asks of a volume of *shape_zyx*, or
    None while it asks none (``none``, or a factor that reduces no axis)."""
    down = downsample_for(downsample_factor(settings.get(name)), shape_zyx, voxel_size_zyx)
    return None if down.is_identity else down


def downsample_volume(
    volume: np.ndarray,
    downsample: Downsample | None,
    *,
    reduce: str,
    description: str = "volume",
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """*volume* reduced as *downsample* says (block *reduce*, see
    :func:`block_reduce_volume`); *volume* itself for None or an identity. A
    volume that is not the shape the reduction was worked out for raises
    ``ValueError``: it cannot be on the same grid afterwards."""
    if downsample is None or downsample.is_identity:
        return volume
    if tuple(volume.shape) != tuple(downsample.source_shape_zyx):
        raise ValueError(
            f"Cannot downsample the {description} of shape {tuple(volume.shape)} as a volume "
            f"of {tuple(downsample.source_shape_zyx)}: the two are not on one grid."
        )
    logger.info("Downsample on import, %s: %s.", description, downsample.describe())
    return block_reduce_volume(
        volume, downsample.axis_factors_zyx, reduce,
        use_memmap=use_memmap, memmap_directory=memmap_directory,
    )


def spread_onto_grid(
    coarse: np.ndarray,
    coarse_blocks: Sequence[int],
    fine_blocks: Sequence[int],
    fine_shape_zyx: Sequence[int],
    *,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """*coarse*, a volume reduced by *coarse_blocks* from some source, spread
    over the grid the same source gives reduced by the smaller *fine_blocks*:
    each fine voxel takes the coarse voxel its block's centre lies in, so the
    result is *fine_shape_zyx* voxels of the coarse image's values."""
    coarse_blocks = tuple(int(b) for b in coarse_blocks)
    fine_blocks = tuple(int(b) for b in fine_blocks)
    fine_shape = tuple(int(n) for n in fine_shape_zyx)
    indices = []
    for n_fine, n_coarse, fine_b, coarse_b in zip(fine_shape, coarse.shape, fine_blocks, coarse_blocks):
        # The fine voxel's block centre, (i + 1/2) * fine_b, in the source's
        # voxels; the coarse voxel it lies in; never past the last one.
        centre_twice = (2 * np.arange(n_fine, dtype=np.int64) + 1) * fine_b
        indices.append(np.minimum(centre_twice // (2 * coarse_b), n_coarse - 1))
    if use_memmap:
        from haemolynx.preprocessing.memmap_support import new_memmap_array, slab_step

        out = new_memmap_array(fine_shape, coarse.dtype, directory=memmap_directory)
        step = slab_step(fine_shape)
        for start in range(0, fine_shape[0], step):
            out[start:start + step] = coarse[np.ix_(indices[0][start:start + step], indices[1], indices[2])]
        return out
    return coarse[np.ix_(*indices)]


def downsample_raw_volume(
    volume: np.ndarray,
    downsample: Downsample | None,
    raw_factor: int,
    *,
    description: str = "raw image",
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """A raw intensity *volume* on the segmented image's source grid, read
    onto the segmentation's grid: reduced by *downsample* (block mean).
    *raw_factor* is ``raw_image_downsample``'s: a finer one than the
    segmentation's changes nothing (the raw image has to lie on the network's
    grid, and it is said so); a coarser one reduces by its own blocks first
    and spreads each coarse voxel over the grid (:func:`spread_onto_grid`).
    *volume* itself when neither reduces anything.
    """
    raw_factor = max(1, int(raw_factor))
    if downsample is None:
        if raw_factor <= 1:
            return volume
        raise ValueError(
            f"raw_image_downsample is {raw_factor}x, but the grid the {description} is read onto "
            "is not given: pass the segmented image's Downsample (an identity when it is not reduced)."
        )
    if downsample.is_identity and raw_factor <= 1:
        return volume
    if tuple(volume.shape) != tuple(downsample.source_shape_zyx):
        raise ValueError(
            f"Cannot downsample the {description} of shape {tuple(volume.shape)} as a volume "
            f"of {tuple(downsample.source_shape_zyx)}: the two are not on one grid."
        )
    raw_blocks = axis_downsample_factors(raw_factor, downsample.source_voxel_size_zyx)
    if all(r <= s for r, s in zip(raw_blocks, downsample.axis_factors_zyx)):
        if raw_factor != downsample.factor:
            logger.info(
                "raw_image_downsample is %dx but the network is on the segmented image's %dx grid, "
                "which the %s must lie on: it is read at %dx.",
                raw_factor, downsample.factor, description, downsample.factor,
            )
        return downsample_volume(
            volume, downsample, reduce="mean", description=description,
            use_memmap=use_memmap, memmap_directory=memmap_directory,
        )
    coarse = block_reduce_volume(
        volume, raw_blocks, "mean", use_memmap=use_memmap, memmap_directory=memmap_directory
    )
    logger.info(
        "Downsample on import, %s: %dx, every %s voxels (x, y, z) become one, then each "
        "spread over the network's %dx grid (%s voxels).",
        description, raw_factor, " x ".join(str(b) for b in reversed(raw_blocks)),
        downsample.factor, " x ".join(str(n) for n in reversed(downsample.shape_zyx)),
    )
    spread = spread_onto_grid(
        coarse, raw_blocks, downsample.axis_factors_zyx, downsample.shape_zyx,
        use_memmap=use_memmap, memmap_directory=memmap_directory,
    )
    if use_memmap and isinstance(coarse, np.memmap):
        from haemolynx.preprocessing.memmap_support import release_memmap_array

        release_memmap_array(coarse)
    return spread


# --------------------------------------------------------------------------
# On the graph
# --------------------------------------------------------------------------


def record_on_graph(
    G: Any,
    downsample: Downsample | None,
    *,
    raw_factor: int = 1,
    image_shape_zyx: Sequence[int] | None = None,
    voxel_size_zyx: Sequence[float] | None = None,
) -> None:
    """Keep on the network *G* what it was built on (:data:`GRAPH_ATTRIBUTE`):
    the segmented image's reduction and the raw image's factor. Nothing is
    kept -- and an old record dropped -- when neither reduces anything. With
    no *downsample*, the record is an identity over *image_shape_zyx* voxels
    of *voxel_size_zyx*, which a raw factor alone then needs."""
    raw_factor = max(1, int(raw_factor))
    if downsample is None or downsample.is_identity:
        if raw_factor <= 1 or image_shape_zyx is None or voxel_size_zyx is None:
            if raw_factor > 1:
                logger.warning(
                    "raw_image_downsample is %dx, but the network's grid is not known here: "
                    "the raw image is read as it is.", raw_factor,
                )
            G.graph.pop(GRAPH_ATTRIBUTE, None)
            return
        downsample = downsample_for(1, image_shape_zyx, voxel_size_zyx)
    record = downsample.to_record()
    record["raw_factor"] = raw_factor
    record["raw_axis_factors_zyx"] = tuple(
        int(b) for b in axis_downsample_factors(raw_factor, downsample.source_voxel_size_zyx)
    )
    G.graph[GRAPH_ATTRIBUTE] = record


def graph_downsample(G: Any) -> Downsample | None:
    """How the network *G*'s segmented image was reduced, from its
    :data:`GRAPH_ATTRIBUTE`; None for one built on the image as read."""
    record = getattr(G, "graph", {}).get(GRAPH_ATTRIBUTE) if G is not None else None
    if not record:
        return None
    down = Downsample.from_record(record)
    return None if down.is_identity else down


def graph_raw_downsample_factor(G: Any) -> int:
    """``raw_image_downsample``'s factor the network *G* recorded; 1 without one."""
    record = getattr(G, "graph", {}).get(GRAPH_ATTRIBUTE) if G is not None else None
    return 1 if not record else max(1, int(record.get("raw_factor", 1)))


def _release_if_replaced(volume: np.ndarray, replaced: np.ndarray, keep: np.ndarray) -> None:
    """*replaced* -- a disk-backed copy this module made on the way from the
    caller's *keep* to *volume* -- is released; never the caller's own."""
    if replaced is not volume and replaced is not keep and isinstance(replaced, np.memmap):
        from haemolynx.preprocessing.memmap_support import release_memmap_array

        release_memmap_array(replaced)


def mask_volume_like_graph(
    volume: np.ndarray,
    G: Any,
    *,
    description: str,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """A mask read whole from disk beside the main image, placed on the
    network *G*'s grid: cut to the ``crop_on_import`` box the network was
    built in (:func:`haemolynx.io.crop.crop_volume_like_graph`), then reduced
    by the network's downsample on import (block max). *volume* itself when
    the network was built on the image as read; a volume of another shape is
    refused as the crop refuses one."""
    cropped = crop_volume_like_graph(
        volume, G, description=description, use_memmap=use_memmap, memmap_directory=memmap_directory
    )
    record = getattr(G, "graph", {}).get(GRAPH_ATTRIBUTE) if G is not None else None
    if not record:
        return cropped
    reduced = downsample_volume(
        cropped, Downsample.from_record(record), reduce="max", description=description,
        use_memmap=use_memmap, memmap_directory=memmap_directory,
    )
    _release_if_replaced(reduced, cropped, volume)
    return reduced


def raw_volume_like_graph(
    volume: np.ndarray,
    G: Any,
    *,
    description: str,
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """A raw intensity image read whole from disk, placed on the network
    *G*'s grid: cut to the network's ``crop_on_import`` box, then read onto its
    grid as :func:`downsample_raw_volume` says, by the segmented image's
    reduction and the raw image's own factor the network recorded."""
    cropped = crop_volume_like_graph(
        volume, G, description=description, use_memmap=use_memmap, memmap_directory=memmap_directory
    )
    record = getattr(G, "graph", {}).get(GRAPH_ATTRIBUTE) if G is not None else None
    if not record:
        return cropped
    reduced = downsample_raw_volume(
        cropped, Downsample.from_record(record), int(record.get("raw_factor", 1)),
        description=description, use_memmap=use_memmap, memmap_directory=memmap_directory,
    )
    _release_if_replaced(reduced, cropped, volume)
    return reduced
