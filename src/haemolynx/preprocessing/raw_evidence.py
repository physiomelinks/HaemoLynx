"""The raw image as evidence for graph building, read where the mask is silent.

The segmentation was made from the raw channel, so inside the mask the raw
image mostly repeats what the mask says: on E14.5, dextran contrast along a
mask-covered chain separated Harvey's "vessel" from "not a vessel" marks no
better than marking everything one way (grid G4, 2026-10-13). The raw image
says something new in two places only, and this module reads it there:

- **background the mask has none in** -- the stretch a bridge or a join
  crosses, the hole inside a short loop, a tunnel through a blob: lumen-bright
  there means a dropout or a pinhole in the segmentation, dark means tissue;
- **inside one blob of mask that is really two lumens**: the raw profile across
  the chord between two centrelines dips at the wall between two vessels and
  does not dip inside one.

Two stains, two readings of the same questions (``raw_image_stain``). A
plasma label (dextran) fills the lumen: a vessel is bright where its
centreline runs, and a wall between two lumens is a dip. An endothelial label
(claudin-5) is the wall: the lumen is dark, a vessel is a bright ring round
the centreline, and a wall between two lumens is a bright line across the
chord. :class:`RawEvidencePlasma` and :class:`RawEvidenceEndothelial` answer
the same three questions -- :meth:`RawEvidence.signal_along`,
:meth:`RawEvidence.wall_between`, :meth:`RawEvidence.hole_is_lumen` -- each
in its stain's own terms, so a rule asks one object and never which stain it
has.

Nothing here reads the raw volume whole. The calibration
(:func:`calibrate_raw`) samples blocks round the mask's vessels, as the stain
reader does; every question reads one window round its own points
(:meth:`RawEvidence._read`, the pattern of
``graph.post_processing.IntensityCostField``), blurred and sampled there, so
a memory-mapped stack stays on disk and the cost of a build is the sum of its
questions, counted in :attr:`RawEvidence.stats`. Brightness is normalised to
the run's own levels -- 0 at the background, 1 at the lumen (plasma) or the
wall (endothelial) -- so every threshold is in those units.

``build_network`` makes the object when the run has a raw image
(:func:`raw_evidence_from_settings`) and keeps its calibration on the graph
(:data:`GRAPH_ATTRIBUTE`); the rules that will read it land one at a time
(``PLAN_raw_confirmation.md``).
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

PLASMA_COLUMN = "plasma_column"
ENDOTHELIAL = "endothelial"
AUTO = "auto"
#: The stains a calibration can name.
STAINS = (PLASMA_COLUMN, ENDOTHELIAL)

#: Where the graph keeps the calibration its build read (``RawCalibration.as_dict``).
GRAPH_ATTRIBUTE = "raw_evidence"

#: The vessel level is read at mask voxels at least this deep in the lumen (um),
#: and at least this share of the deepest voxel in the block, so a thick wall's
#: inner layer is not read as lumen.
CORE_DEPTH_UM = 1.5
CORE_DEPTH_FRACTION = 0.5
#: The wall level is read within this of the mask's surface, either side (um).
WALL_REACH_UM = 1.0
#: The background level is read at least this far from the mask (um).
BACKGROUND_FROM_UM = 4.0
#: How many blocks of mask the calibration samples, and their size (um).
CALIBRATION_BLOCKS = 16
CALIBRATION_BLOCK_UM = 40.0
#: Read round each block, so a vessel the block cuts keeps its true depth.
CALIBRATION_MARGIN_UM = 6.0
#: ``auto`` names the stain by the share of the signal above the background
#: the lumen core carries, against the brighter of the two sides of the mask's
#: surface: at least PLASMA_CORE_SHARE of it and the lumen is the label, at
#: most ENDOTHELIAL_CORE_SHARE and the wall is; between, it cannot tell. A
#: capillary bed thinner than the image's blur reads as lumen either way, so
#: ``raw_image_stain`` can say instead.
PLASMA_CORE_SHARE = 0.8
ENDOTHELIAL_CORE_SHARE = 0.5
#: The level a reading is normalised by must stand this many robust spreads
#: of the background above it, or there is nothing to read.
SIGNAL_OVER_SPREAD = 1.0
#: Gaussian blur (voxels) of each window before sampling, so a question is
#: not answered by one speck of shot noise: E14.5's dextran channel has a
#: median of 3 counts inside the mask and 0 outside, and only once blurred by
#: a voxel do 98% of lumen voxels stand above the background's 99th
#: percentile. The calibration reads its levels on blocks blurred the same
#: way, so a wall thinner than the blur still normalises to 1.
WINDOW_SMOOTHING_SIGMA_VOX = 1.0
#: The lumen core is never asked to lie deeper than this (um): the middle of
#: a thick vessel is dim under a plasma label, and is not its level.
CORE_DEPTH_MAX_UM = 3.0
#: A sample counts as vessel at or above this normalised brightness.
BRIGHT_THRESHOLD = 0.5
#: Paths and chords are sampled this many voxels apart (of the finest axis).
SAMPLE_STEP_VOXELS = 0.5
#: A question's points are read in windows of at most this many voxels: a
#: long path is read in pieces rather than as one box round all of it.
MAX_WINDOW_VOXELS = 2_000_000


@dataclass(frozen=True)
class RawCalibration:
    """The run's own levels, read once from the raw image round the mask."""

    stain: str
    #: Median raw value deep in the segmented vessels.
    vessel: float
    #: Median raw value within :data:`WALL_REACH_UM` of the mask's surface.
    wall: float
    #: Median raw value well away from the mask.
    background: float
    #: The background's robust spread (1.4826 x its median absolute deviation).
    background_spread: float
    #: ``(vessel - wall) / (|vessel| + |wall|)`` above the background.
    contrast: float
    blocks: int
    #: The blur (voxels) the levels were read under; the questions use the same.
    sigma_vox: float = WINDOW_SMOOTHING_SIGMA_VOX

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "RawCalibration":
        return cls(
            stain=str(record["stain"]),
            vessel=float(record["vessel"]),
            wall=float(record["wall"]),
            background=float(record["background"]),
            background_spread=float(record["background_spread"]),
            contrast=float(record.get("contrast", 0.0)),
            blocks=int(record.get("blocks", 0)),
            sigma_vox=float(record.get("sigma_vox", WINDOW_SMOOTHING_SIGMA_VOX)),
        )

    @classmethod
    def from_stain_reading(cls, reading: Any, stain: Optional[str] = None) -> "RawCalibration":
        """From a ``preprocessing.raw_stain.StainReading`` (duck-typed: ``core``,
        ``wall``, ``background``, ``background_spread``, ``contrast``,
        ``blocks``, ``stain``), so a run that has read the stain already does
        not sample the image twice. *stain* overrides the reading's."""
        named = stain if stain in STAINS else getattr(reading, "stain", None)
        if named not in STAINS:
            raise ValueError(f"The stain reading names no stain ({named!r}); pass stain=")
        return cls(
            stain=str(named),
            vessel=float(reading.core),
            wall=float(reading.wall),
            background=float(reading.background),
            background_spread=float(reading.background_spread),
            contrast=float(getattr(reading, "contrast", 0.0)),
            blocks=int(getattr(reading, "blocks", 0)),
        )

    @property
    def level(self) -> float:
        """The raw value normalised brightness 1 stands for: the lumen's for a
        plasma label, the wall's for an endothelial one."""
        return self.vessel if self.stain == PLASMA_COLUMN else self.wall

    def describe(self) -> str:
        what = {PLASMA_COLUMN: "plasma (lumen) label", ENDOTHELIAL: "endothelial (wall) label"}
        return (
            f"{what.get(self.stain, self.stain)}: lumen {self.vessel:.4g}, wall {self.wall:.4g}, "
            f"background {self.background:.4g} (spread {self.background_spread:.3g}), "
            f"contrast {self.contrast:+.2f}, {self.blocks} blocks"
        )


@dataclass
class RawEvidenceStats:
    """What a run's questions cost: how many windows were read, how many
    voxels they held, and the seconds spent reading and sampling them."""

    queries: int = 0
    voxels_read: int = 0
    seconds: float = 0.0


@dataclass(frozen=True)
class RawSignal:
    """:meth:`RawEvidence.signal_along`'s answer for one path."""

    #: Share of the path's samples that read as vessel (:data:`BRIGHT_THRESHOLD`).
    fraction_bright: float
    #: The longest run of samples that do not, in microns.
    longest_dark_um: float
    samples: int


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def _sample_mask_voxels(mask: np.ndarray, rng: np.random.Generator, *, count: int) -> list[np.ndarray]:
    """Up to *count* mask voxels spread over random slices, read one slice at
    a time so a memory-mapped mask is never pulled in whole."""
    depth = int(mask.shape[0])
    if depth == 0:
        return []
    picks: list[np.ndarray] = []
    for z in rng.permutation(depth)[: max(1, min(depth, count))]:
        where = np.argwhere(np.asarray(mask[int(z)], dtype=bool))
        if len(where):
            row = where[rng.integers(len(where))]
            picks.append(np.array([int(z), int(row[0]), int(row[1])]))
        if len(picks) >= count:
            break
    return picks


def calibrate_raw(
    raw: Any,
    mask: np.ndarray,
    voxel_size_zyx: Sequence[float],
    *,
    stain: str = AUTO,
    blocks: int = CALIBRATION_BLOCKS,
    block_um: float = CALIBRATION_BLOCK_UM,
    seed: int = 0,
    sigma_vox: float = WINDOW_SMOOTHING_SIGMA_VOX,
) -> Optional[RawCalibration]:
    """The run's lumen, wall and background levels, read in *blocks* blocks of
    *block_um* round sampled mask voxels (each block its own distance
    transforms, as ``raw_stain.read_raw_stain`` does), and the stain: *stain*
    as given, or with ``auto`` from which of the lumen and the wall stands
    higher. None, with a log line saying why, when the image has nothing to
    read: no block with lumen, wall and background in it, or the level the
    stain is read at not standing :data:`SIGNAL_OVER_SPREAD` spreads above the
    background. *raw* and *mask* share one grid; either may be a memory map."""
    from scipy import ndimage

    if stain not in (AUTO, *STAINS):
        raise ValueError(f"stain must be one of {(AUTO, *STAINS)}, got {stain!r}")
    vs = np.asarray(voxel_size_zyx, dtype=float)
    shape = np.asarray(mask.shape)
    if tuple(np.shape(raw)) != tuple(mask.shape):
        raise ValueError(
            f"The raw image {tuple(np.shape(raw))} and the mask {tuple(mask.shape)} differ in shape"
        )
    half = np.maximum(np.ceil(block_um / 2.0 / vs).astype(int), 2)
    margin = np.ceil(CALIBRATION_MARGIN_UM / vs).astype(int)
    rng = np.random.default_rng(seed)
    core_values, wall_in_values, wall_out_values, background_values = [], [], [], []
    used = 0
    for centre in _sample_mask_voxels(mask, rng, count=blocks * 8):
        if used >= blocks:
            break
        lo = np.maximum(centre - half - margin, 0)
        hi = np.minimum(centre + half + margin + 1, shape)
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        inner = np.zeros(tuple(hi - lo), dtype=bool)
        inner[tuple(slice(int(max(c - h - a, 0)), int(c + h + 1 - a)) for c, h, a in zip(centre, half, lo))] = True
        vessel = np.asarray(mask[box], dtype=bool)
        if not vessel.any() or vessel.all():
            continue
        depth = ndimage.distance_transform_edt(vessel, sampling=vs)
        in_vessel = inner & vessel
        if not in_vessel.any():
            continue
        core = in_vessel & (
            depth >= max(CORE_DEPTH_UM, min(CORE_DEPTH_FRACTION * float(depth[in_vessel].max()), CORE_DEPTH_MAX_UM))
        )
        outside = ndimage.distance_transform_edt(~vessel, sampling=vs)
        # The wall sits just inside the mask's surface (a segmentation of the
        # wall itself) or just outside it (a lumen segmentation under a wall
        # label), so each side of the surface is read on its own.
        wall_in = in_vessel & (depth <= WALL_REACH_UM)
        wall_out = inner & ~vessel & (outside <= WALL_REACH_UM)
        background = inner & ~vessel & (outside >= BACKGROUND_FROM_UM)
        if core.sum() < 8 or (wall_in.sum() < 8 and wall_out.sum() < 8) or background.sum() < 8:
            continue
        # Read as the questions read: blurred, with the margin as context.
        values = np.asarray(raw[box], dtype=np.float32)
        if sigma_vox > 0:
            values = ndimage.gaussian_filter(values, sigma_vox)
        values = values.astype(np.float64)
        core_values.append(values[core])
        wall_in_values.append(values[wall_in])
        wall_out_values.append(values[wall_out])
        background_values.append(values[background])
        used += 1
    if not used:
        logger.info("Raw evidence: no block of the mask held lumen, wall and background together; not read.")
        return None
    core_all, background_all = np.concatenate(core_values), np.concatenate(background_values)
    background_level = float(np.median(background_all))
    spread = float(1.4826 * np.median(np.abs(background_all - background_level)))
    core_level = float(np.median(core_all))
    wall_level = max(
        float(np.median(np.concatenate(side))) if any(len(v) for v in side) else -np.inf
        for side in (wall_in_values, wall_out_values)
    )
    lift_core, lift_wall = core_level - background_level, wall_level - background_level
    scale = abs(lift_core) + abs(lift_wall)
    contrast = float((lift_core - lift_wall) / scale) if scale > 0 else 0.0
    if stain == AUTO:
        top = max(lift_core, lift_wall)
        share = lift_core / top if top > 0 else 0.0
        if top <= SIGNAL_OVER_SPREAD * max(spread, 0.0):
            logger.info(
                "Raw evidence: neither the lumen (%.4g) nor the wall (%.4g) stands above the background "
                "(%.4g, spread %.3g); not read.",
                core_level, wall_level, background_level, spread,
            )
            return None
        if share >= PLASMA_CORE_SHARE:
            stain = PLASMA_COLUMN
        elif share <= ENDOTHELIAL_CORE_SHARE:
            stain = ENDOTHELIAL
        else:
            logger.info(
                "Raw evidence: the lumen (%.4g) carries %.0f%% of the wall's (%.4g) signal above the "
                "background (%.4g), between a lumen and a wall label, so the stain cannot be told; "
                "set raw_image_stain. Not read.",
                core_level, 100 * share, wall_level, background_level,
            )
            return None
    calibration = RawCalibration(
        stain=stain, vessel=core_level, wall=wall_level, background=background_level,
        background_spread=spread, contrast=contrast, blocks=used, sigma_vox=float(sigma_vox),
    )
    if calibration.level - background_level <= SIGNAL_OVER_SPREAD * max(spread, 0.0):
        logger.info(
            "Raw evidence: the %s level (%.4g) does not stand above the background (%.4g, spread %.3g); not read.",
            "lumen" if stain == PLASMA_COLUMN else "wall", calibration.level, background_level, spread,
        )
        return None
    return calibration


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _along(path_um: np.ndarray, step_um: float) -> np.ndarray:
    """*path_um* resampled every *step_um* along it, both ends kept."""
    path = np.asarray(path_um, dtype=float).reshape(-1, 3)
    if len(path) < 2:
        return path
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        length = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(length / step_um)))
        for t in np.linspace(0.0, 1.0, n + 1)[1:]:
            out.append(a + t * (b - a))
    return np.asarray(out)


def _tangents(samples: np.ndarray) -> np.ndarray:
    """A unit direction at each point of an ordered path (central differences)."""
    if len(samples) < 2:
        return np.tile([0.0, 0.0, 1.0], (len(samples), 1))
    tangents = np.gradient(np.asarray(samples, dtype=float), axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    norms[norms == 0] = 1.0
    return tangents / norms[:, None]


def _across(tangents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors across each tangent, spanning the plane normal to it."""
    tangents = np.asarray(tangents, dtype=float).reshape(-1, 3)
    helper = np.where(np.abs(tangents[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    p = np.cross(tangents, helper)
    p /= np.maximum(np.linalg.norm(p, axis=1), 1e-12)[:, None]
    q = np.cross(tangents, p)
    return p, q


#: Directions round a point an endothelial wall is looked for in, and the
#: radii, about the vessel's own, it is looked for at.
_RING_DIRECTIONS = 8
_RING_RADII_UM = (-WALL_REACH_UM, 0.0, WALL_REACH_UM)


def _longest_run(flags: np.ndarray) -> int:
    longest = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest


def _inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd test of 2-D *points* against the closed *polygon*."""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        crosses = (y0 > y) != (y1 > y)
        if not crosses.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < x_at)
    return inside


def _distance_to_sides(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Distance from each 2-D point to the nearest side of the closed *polygon*."""
    best = np.full(len(points), np.inf)
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        ab = b - a
        length2 = float(ab @ ab)
        t = np.zeros(len(points)) if length2 == 0 else np.clip(((points - a) @ ab) / length2, 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(points - (a + t[:, None] * ab), axis=1))
    return best


def _loop_plane(loop_um: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The loop's own plane: ``(centroid, u, v, polygon)`` with the loop's
    vertices as 2-D *polygon* coordinates on the in-plane axes *u*, *v*."""
    points = np.asarray(loop_um, dtype=float).reshape(-1, 3)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    u, v = vt[0], vt[1] if len(vt) > 1 else np.zeros(3)
    polygon = (points - centroid) @ np.vstack([u, v]).T
    return centroid, u, v, polygon


# --------------------------------------------------------------------------
# The evidence
# --------------------------------------------------------------------------


class RawEvidence:
    """One raw channel, read the ways a graph-building rule asks about it.

    *raw* is voxel-indexed on the run's own grid, canonical ``(z, y, x)``, the
    same shape as the mask the graph was built on (a cropped run passes the
    raw image cut to its box: :func:`make_raw_evidence`); a memory map, an
    array, or anything sliceable with ``shape``. Points are physical microns
    on *voxel_size_zyx*. Use the stain's subclass, or :func:`make_raw_evidence`.
    """

    stain: str = ""

    def __init__(
        self,
        raw: Any,
        voxel_size_zyx: Sequence[float],
        calibration: RawCalibration,
        *,
        sigma_vox: Optional[float] = None,
    ):
        shape = tuple(int(n) for n in np.shape(raw))
        if len(shape) != 3:
            raise ValueError(f"The raw image must be a 3D (z, y, x) volume, got shape {shape}")
        self.raw = raw
        self.shape = shape
        self.voxel_size_zyx = tuple(float(v) for v in voxel_size_zyx)
        self.calibration = calibration
        # The blur the levels were read under, unless a caller insists.
        self.sigma_vox = float(calibration.sigma_vox if sigma_vox is None else sigma_vox)
        self.stats = RawEvidenceStats()
        self._spacing = np.asarray(self.voxel_size_zyx, dtype=float)
        self._step_um = SAMPLE_STEP_VOXELS * float(self._spacing.min())
        self._scale = float(calibration.level - calibration.background)
        if not self._scale > 0:
            raise ValueError(
                f"The calibration's {calibration.stain} level ({calibration.level:.4g}) does not "
                f"stand above its background ({calibration.background:.4g})"
            )

    # -- reading ---------------------------------------------------------

    def _read(self, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The blurred window ``[lo, hi)`` (voxel indices, clipped to the
        image), padded by the blur's reach so it has no edge, as float32,
        with the index its corner sits at."""
        pad = int(np.ceil(3.0 * self.sigma_vox)) if self.sigma_vox > 0 else 0
        plo = np.maximum(np.asarray(lo, dtype=int) - pad, 0)
        phi = np.minimum(np.asarray(hi, dtype=int) + pad, np.asarray(self.shape))
        if np.any(phi <= plo):
            return np.zeros((0, 0, 0), dtype=np.float32), plo
        crop = np.asarray(self.raw[tuple(slice(int(a), int(b)) for a, b in zip(plo, phi))], dtype=np.float32)
        self.stats.queries += 1
        self.stats.voxels_read += int(crop.size)
        if self.sigma_vox > 0 and crop.size:
            from scipy.ndimage import gaussian_filter

            crop = gaussian_filter(crop, self.sigma_vox)
        return crop, plo

    def _chunks(self, index: np.ndarray, reach: np.ndarray):
        """Runs of consecutive points whose window, *reach* voxels round them,
        stays under :data:`MAX_WINDOW_VOXELS`."""
        start = 0
        lo = hi = None
        for i, point in enumerate(index):
            p_lo, p_hi = np.floor(point) - reach, np.ceil(point) + reach + 1
            n_lo = p_lo if lo is None else np.minimum(lo, p_lo)
            n_hi = p_hi if hi is None else np.maximum(hi, p_hi)
            if lo is not None and float(np.prod(n_hi - n_lo)) > MAX_WINDOW_VOXELS:
                yield start, i, lo, hi
                start, lo, hi = i, p_lo, p_hi
            else:
                lo, hi = n_lo, n_hi
        if lo is not None:
            yield start, len(index), lo, hi

    def _values(self, points_um: np.ndarray, *, dilate_vox: Optional[np.ndarray] = None) -> np.ndarray:
        """The blurred raw value at each point (trilinear), the window first
        grey-dilated by a box of *dilate_vox* half-sides when given; the
        background level at a point outside the image."""
        from scipy.ndimage import map_coordinates, maximum_filter

        started = time.perf_counter()
        points = np.asarray(points_um, dtype=float).reshape(-1, 3)
        out = np.full(len(points), self.calibration.background, dtype=float)
        if not len(points):
            return out
        index = points / self._spacing
        shape = np.asarray(self.shape)
        inside = np.all(index >= -0.5, axis=1) & np.all(index <= shape - 0.5, axis=1)
        reach = np.ones(3, dtype=int) if dilate_vox is None else np.asarray(dilate_vox, dtype=int) + 1
        for start, stop, lo, hi in self._chunks(index, reach):
            crop, plo = self._read(np.maximum(lo, 0).astype(int), hi.astype(int))
            if not crop.size:
                continue
            if dilate_vox is not None:
                crop = maximum_filter(crop, size=tuple(int(2 * d + 1) for d in dilate_vox), mode="nearest")
            local = (index[start:stop] - plo).T
            out[start:stop] = map_coordinates(crop, local, order=1, mode="nearest")
        out[~inside] = self.calibration.background
        self.stats.seconds += time.perf_counter() - started
        return out

    def brightness(self, points_um: np.ndarray) -> np.ndarray:
        """Normalised brightness at each point: 0 at the background level, 1 at
        the stain's own level (lumen or wall), clipped to [0, 1]."""
        return self._normalise(self._values(points_um))

    def _normalise(self, values: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(values, dtype=float) - self.calibration.background) / self._scale, 0.0, 1.0)

    # -- the questions --------------------------------------------------

    def vesselness(
        self, points_um: np.ndarray, radius_um: float, tangents: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Evidence, 0 to 1, that a vessel of about *radius_um* runs through
        each point: its own brightness for a plasma label, its wall's for an
        endothelial one -- read round the point in the plane across
        *tangents* (one unit vector per point, the vessel's direction there)
        when given, else all round it."""
        raise NotImplementedError

    def signal_along(self, path_um: np.ndarray, radius_um: float) -> RawSignal:
        """How much of *path_um* the raw image shows vessel along, sampled every
        half voxel: the raw analogue of ``BridgeMaskSupport``'s mask fraction
        and longest background gap."""
        samples = _along(path_um, self._step_um)
        if len(samples) == 0:
            return RawSignal(0.0, 0.0, 0)
        bright = self.vesselness(samples, radius_um, _tangents(samples)) >= BRIGHT_THRESHOLD
        if len(samples) == 1:
            return RawSignal(float(bright[0]), 0.0 if bright[0] else self._step_um, 1)
        steps = np.linalg.norm(np.diff(samples, axis=0), axis=1)
        step = float(steps.mean()) if len(steps) else self._step_um
        return RawSignal(float(bright.mean()), float(_longest_run(~bright) * step), int(len(samples)))

    def wall_between(self, a_um: Sequence[float], b_um: Sequence[float]) -> float:
        """Evidence, 0 to 1, of a wall between two centreline points: how far
        the raw profile along the chord dips below its ends for a plasma
        label, how far it rises above them for an endothelial one. 0 for a
        chord whose ends are not in a vessel at all."""
        raise NotImplementedError

    def _chord(self, a_um, b_um) -> np.ndarray:
        """The raw profile along the chord, above the background, in units of
        the stain's level -- unclipped, so a vessel brighter than the run's
        level (a wide one, or one near the top of the stack) keeps its dip."""
        a = np.asarray(a_um, dtype=float).reshape(3)
        b = np.asarray(b_um, dtype=float).reshape(3)
        values = self._values(_along(np.vstack([a, b]), self._step_um))
        return (values - self.calibration.background) / self._scale

    def _own_level(self, loop_um: np.ndarray) -> float:
        """The loop's own vessels' brightness (median along it, unclipped)."""
        samples = _along(np.asarray(loop_um, dtype=float).reshape(-1, 3), self._step_um)
        return float(np.median((self._values(samples) - self.calibration.background) / self._scale))

    def hole_interior(self, loop_um: np.ndarray, radius_um: float) -> tuple[np.ndarray, np.ndarray]:
        """Points filling the loop's hole in the loop's own plane, one per
        finest voxel, at least *radius_um* in from its sides (the vessels'
        own lumen left out), and each one's distance to the nearest side."""
        centroid, u, v, polygon = _loop_plane(loop_um)
        if len(polygon) < 3:
            return np.zeros((0, 3)), np.zeros(0)
        step = float(self._spacing.min())
        lo, hi = polygon.min(axis=0), polygon.max(axis=0)
        xs = np.arange(lo[0], hi[0] + step, step)
        ys = np.arange(lo[1], hi[1] + step, step)
        grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
        inside = _inside_polygon(grid, polygon)
        grid = grid[inside]
        if not len(grid):
            return np.zeros((0, 3)), np.zeros(0)
        distance = _distance_to_sides(grid, polygon)
        keep = distance >= radius_um
        points = centroid + grid[keep, :1] * u + grid[keep, 1:] * v
        return points, distance[keep]

    def hole_is_lumen(self, loop_um: np.ndarray, radius_um: float) -> float:
        """Evidence, 0 to 1, that the hole a loop runs round is lumen the
        segmentation missed (a pinhole, so a false loop) rather than tissue
        (a real one): the hole's own brightness for a plasma label, the
        absence of a wall lining it for an endothelial one. NaN when the hole
        is too small to hold a point clear of the loop's own vessels."""
        raise NotImplementedError


class RawEvidencePlasma(RawEvidence):
    """A plasma (luminal) label: the lumen is bright, the tissue dark."""

    stain = PLASMA_COLUMN

    def vesselness(
        self, points_um: np.ndarray, radius_um: float, tangents: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.brightness(points_um)

    def wall_between(self, a_um, b_um) -> float:
        profile = self._chord(a_um, b_um)
        if len(profile) < 3:
            return 0.0
        ends = float(min(profile[0], profile[-1]))
        if ends <= 0.0:
            return 0.0
        return float(np.clip((ends - profile[1:-1].min()) / ends, 0.0, 1.0))

    def hole_is_lumen(self, loop_um, radius_um: float) -> float:
        points, _ = self.hole_interior(loop_um, radius_um)
        if not len(points):
            return float("nan")
        # Against the loop's own vessels, not the run's level: a loop on a
        # wide bright vessel has a bright hole only if it is as bright as they are.
        own = max(self._own_level(loop_um), BRIGHT_THRESHOLD)
        hole = np.median((self._values(points) - self.calibration.background) / self._scale)
        return float(np.clip(hole / own, 0.0, 1.0))


class RawEvidenceEndothelial(RawEvidence):
    """An endothelial (wall) label: the wall is bright, the lumen and the
    tissue dark. A vessel at a point is a bright wall within the vessel's
    radius of it, read by grey-dilating the window by that reach; a wall
    between two lumens is a bright line across the chord. A capillary thinner
    than the image's blur reads as one bright line, as a plasma label would,
    which the grids that calibrate each rule must allow for."""

    stain = ENDOTHELIAL

    def _reach_vox(self, radius_um: float) -> np.ndarray:
        return np.ceil((float(radius_um) + WALL_REACH_UM) / self._spacing).astype(int)

    def vesselness(
        self, points_um: np.ndarray, radius_um: float, tangents: Optional[np.ndarray] = None
    ) -> np.ndarray:
        points = np.asarray(points_um, dtype=float).reshape(-1, 3)
        if tangents is None:
            return self._normalise(self._values(points, dilate_vox=self._reach_vox(radius_um)))
        # The brightest point on rings across the vessel round each point, at
        # its own radius and a wall's reach either side: the wall, if the
        # vessel is there. Read across, never along, so a gap in the wall
        # along the path is not filled by the wall either side of it.
        p, q = _across(tangents)
        angles = np.linspace(0.0, 2.0 * np.pi, _RING_DIRECTIONS, endpoint=False)
        offsets = []
        for extra in _RING_RADII_UM:
            r = max(float(radius_um) + extra, 0.5 * float(self._spacing.min()))
            for angle in angles:
                offsets.append(r * (np.cos(angle) * p + np.sin(angle) * q))
        ring = points[:, None, :] + np.stack(offsets, axis=1)  # (points, ring, 3)
        values = self._values(ring.reshape(-1, 3)).reshape(len(points), -1)
        return self._normalise(values.max(axis=1))

    def wall_between(self, a_um, b_um) -> float:
        profile = self._chord(a_um, b_um)
        if len(profile) < 3:
            return 0.0
        ends = float(max(profile[0], profile[-1]))
        return float(np.clip(profile[1:-1].max() - ends, 0.0, 1.0))

    def hole_is_lumen(self, loop_um, radius_um: float) -> float:
        points, distance = self.hole_interior(loop_um, radius_um)
        band = distance <= radius_um + 2.0 * WALL_REACH_UM
        if not band.any():
            return float("nan")
        return float(1.0 - np.median(self.brightness(points[band])))


_CLASSES = {PLASMA_COLUMN: RawEvidencePlasma, ENDOTHELIAL: RawEvidenceEndothelial}


def make_raw_evidence(
    raw: Any,
    mask_shape: Sequence[int],
    voxel_size_zyx: Sequence[float],
    calibration: RawCalibration,
    *,
    box: Any = None,
) -> RawEvidence:
    """The stain's :class:`RawEvidence` over *raw*, cut to *box* (an
    ``io.crop.CropBox``, the run's crop on import) as a view -- never a copy,
    so a memory map stays on disk. *raw* must be the whole image the box was
    cut from, or without a box the mask's own shape; anything else is refused
    as a misaligned mask is."""
    shape = tuple(int(n) for n in np.shape(raw))
    expected = tuple(int(n) for n in (box.source_shape_zyx if box is not None else mask_shape))
    if shape != expected:
        raise ValueError(
            f"The raw image's shape {shape} is not the {'image' if box is not None else 'mask'}'s {expected}"
        )
    if box is not None:
        if tuple(int(n) for n in box.shape_zyx) != tuple(int(n) for n in mask_shape):
            raise ValueError(f"The crop box {box.shape_zyx} is not the mask's shape {tuple(mask_shape)}")
        if not box.is_whole_volume:
            raw = raw[box.slices]
    cls = _CLASSES.get(calibration.stain)
    if cls is None:
        raise ValueError(f"Unknown stain {calibration.stain!r}; expected one of {STAINS}")
    return cls(raw, voxel_size_zyx, calibration)


# --------------------------------------------------------------------------
# From a run's settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RawImageSpec:
    """Which raw image a run reads, and what it stains."""

    path: Path
    channel: Optional[int]
    stain: str
    #: The setting the path came from.
    setting: str


def raw_image_for_build(settings: Mapping[str, Any]) -> Optional[RawImageSpec]:
    """The raw image a run's settings name, by either naming: the one raw
    image of the Input tab (``raw_image_path`` / ``raw_image_channel`` /
    ``raw_image_stain``) first, else the FWHM raw image
    (``fwhm_raw_tiff_path`` / ``fwhm_raw_channel``, read as ``auto``: usable
    with FWHM measurement off, as Check segmented image reads it), else the
    endothelial image (``endothelial_image_path`` / ``endothelial_channel``).
    None when the run names no raw image."""
    for path_name, channel_name, stain_name, stain in (
        ("raw_image_path", "raw_image_channel", "raw_image_stain", AUTO),
        ("fwhm_raw_tiff_path", "fwhm_raw_channel", None, AUTO),
        ("endothelial_image_path", "endothelial_channel", None, ENDOTHELIAL),
    ):
        path = settings.get(path_name)
        if path in (None, ""):
            continue
        channel = settings.get(channel_name)
        named = settings.get(stain_name) if stain_name else None
        if named in STAINS:
            stain = str(named)
        return RawImageSpec(Path(path), None if channel in (None, "") else int(channel), stain, path_name)
    return None


def load_raw_channel(
    path: str | Path,
    channel: Optional[int],
    *,
    axis_order: str = "zyx",
    use_memmap: bool = False,
    memmap_directory: str | Path | None = None,
) -> np.ndarray:
    """One channel of a raw TIFF in canonical order. In RAM it keeps the
    file's own dtype (uint16: two bytes a voxel); with *use_memmap* it is the
    float32 disk-backed array ``load_single_channel_tiff_volume`` makes."""
    from haemolynx.haemodynamics.automated import _read_tiff_channel, load_single_channel_tiff_volume
    from haemolynx.io.axis_order import apply_axis_order

    if use_memmap:
        return load_single_channel_tiff_volume(
            path, axis_order=axis_order, use_memmap=True, memmap_directory=memmap_directory, channel=channel,
        )
    return apply_axis_order(_read_tiff_channel(Path(path), 0 if channel is None else int(channel)), axis_order)


def raw_evidence_from_settings(
    settings: Mapping[str, Any],
    mask: np.ndarray,
    voxel_size_zyx: Sequence[float],
    *,
    box: Any = None,
) -> Optional[RawEvidence]:
    """The :class:`RawEvidence` a run's settings give: its raw image loaded
    (:func:`load_raw_channel`), cut to the run's crop *box*, calibrated round
    *mask* (:func:`calibrate_raw`). None when the run names no raw image or
    the image has nothing to read; a misaligned image is refused."""
    spec = raw_image_for_build(settings)
    if spec is None:
        return None
    started = time.perf_counter()
    raw = load_raw_channel(
        spec.path,
        spec.channel,
        axis_order=str(settings.get("image_axis_order") or "zyx"),
        use_memmap=bool(settings.get("use_memmap_loading")),
        memmap_directory=settings.get("memmap_directory"),
    )
    shape = tuple(int(n) for n in np.shape(raw))
    expected = tuple(int(n) for n in (box.source_shape_zyx if box is not None else mask.shape))
    if shape != expected:
        raise ValueError(
            f"The raw image {spec.path.name} ({spec.setting}) has shape {shape}, not the image's "
            f"{expected}: it must be the same stack the segmentation was made from"
        )
    framed = raw[box.slices] if box is not None and not box.is_whole_volume else raw
    calibration = calibrate_raw(framed, mask, voxel_size_zyx, stain=spec.stain)
    if calibration is None:
        return None
    evidence = make_raw_evidence(raw, mask.shape, voxel_size_zyx, calibration, box=box)
    logger.info(
        "Raw evidence from %s%s (%s): %s; calibrated in %.1f s.",
        spec.path.name,
        "" if spec.channel is None else f" channel {spec.channel}",
        spec.setting,
        calibration.describe(),
        time.perf_counter() - started,
    )
    return evidence


__all__ = [
    "AUTO",
    "BRIGHT_THRESHOLD",
    "ENDOTHELIAL",
    "GRAPH_ATTRIBUTE",
    "PLASMA_COLUMN",
    "STAINS",
    "RawCalibration",
    "RawEvidence",
    "RawEvidenceEndothelial",
    "RawEvidencePlasma",
    "RawEvidenceStats",
    "RawImageSpec",
    "RawSignal",
    "calibrate_raw",
    "load_raw_channel",
    "make_raw_evidence",
    "raw_evidence_from_settings",
    "raw_image_for_build",
]
