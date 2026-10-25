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
(:data:`GRAPH_ATTRIBUTE`). No rule reads it: on every marked grid it was
tried on -- one lumen or two (H, H2), a vessel across a gap (M, M2), a loop's
hole (I, I2) -- no raw measure came near the 90 % a rule needs, and most did
no better than marking everything one way, so each of those decisions stays
the mask's (``PLAN_raw_confirmation.md``). A rule that comes to read it takes
the object as an argument and, given None, does what it does today.

An endothelial label also answers what a dead end raises
(``PLAN_dead_ends_raw.md``), across the chain rather than along it: whether a
wall ring runs round the chain, and whether it is the chain's own or a
neighbour's (:meth:`RawEvidenceEndothelial.ring_along`, E1); whether walls
enclose a point, in a lumen another centreline shares
(:meth:`RawEvidenceEndothelial.inside_walls`, E2); whether the walls run on
past the tip, into another vessel or out of the image, along a straight line
(:meth:`RawEvidenceEndothelial.walls_ahead`, E3) or followed round the
vessel's turns (:meth:`RawEvidenceEndothelial.route_ahead`, E4). They measure
only: no rule reads them until a marked grid says what they are worth.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, replace
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
#: ``auto`` names the stain the one way every reader does: ``raw_stain.read_raw_stain``
#: (``raw_image_stain``'s own rule, calibrated on E14.5), so the stain the
#: build reads its evidence under is the one the Diameters stage measures by.
#: A capillary bed thinner than the image's blur reads as lumen either way,
#: so ``raw_image_stain`` can say instead.
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
    #: How the stain was told from the image when it was ``auto``
    #: (``raw_stain.StainReading.as_dict``); None when the run named it.
    reading: Optional[dict] = None
    #: What the image's own wall rings say (``endothelial.calibrate_from_rings``),
    #: when a run has paid for them (:func:`ring_calibration`, 90-120 s on
    #: E14.5): the blur, Gaussian sigma in microns along ``(z, y, x)``; the
    #: wall's thickness in microns; the median fitted wall brightness above
    #: the background, before the blur. None until then: the block levels
    #: above are the default, and :meth:`RawEvidenceEndothelial.ring_along`
    #: fits its rings' blur and wall free without them.
    ring_psf_sigma_zyx: Optional[tuple[float, float, float]] = None
    ring_wall_um: Optional[float] = None
    ring_wall_level: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "RawCalibration":
        """From :meth:`as_dict`'s record, including one a graph saved before
        the ring fields existed (they read as None)."""
        reading = record.get("reading")
        psf = record.get("ring_psf_sigma_zyx")
        wall_um, wall_level = record.get("ring_wall_um"), record.get("ring_wall_level")
        return cls(
            stain=str(record["stain"]),
            vessel=float(record["vessel"]),
            wall=float(record["wall"]),
            background=float(record["background"]),
            background_spread=float(record["background_spread"]),
            contrast=float(record.get("contrast", 0.0)),
            blocks=int(record.get("blocks", 0)),
            sigma_vox=float(record.get("sigma_vox", WINDOW_SMOOTHING_SIGMA_VOX)),
            reading=dict(reading) if reading is not None else None,
            ring_psf_sigma_zyx=None if psf is None else tuple(float(v) for v in psf),
            ring_wall_um=None if wall_um is None else float(wall_um),
            ring_wall_level=None if wall_level is None else float(wall_level),
        )

    def with_rings(
        self,
        psf_sigma_zyx: Sequence[float],
        wall_um: float,
        wall_level: Optional[float] = None,
    ) -> "RawCalibration":
        """This calibration with the ring-fitted blur, wall and wall level."""
        return replace(
            self,
            ring_psf_sigma_zyx=tuple(float(v) for v in psf_sigma_zyx),
            ring_wall_um=float(wall_um),
            ring_wall_level=None if wall_level is None else float(wall_level),
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
        rings = ""
        if self.ring_psf_sigma_zyx is not None and self.ring_wall_um is not None:
            sz, sy, sx = self.ring_psf_sigma_zyx
            rings = f"; rings: blur {sz:.2f}/{sy:.2f}/{sx:.2f} um (z/y/x), wall {self.ring_wall_um:.2f} um"
        return (
            f"{what.get(self.stain, self.stain)}: lumen {self.vessel:.4g}, wall {self.wall:.4g}, "
            f"background {self.background:.4g} (spread {self.background_spread:.3g}), "
            f"contrast {self.contrast:+.2f}, {self.blocks} blocks{rings}"
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
# Dead-end evidence under an endothelial label (PLAN_dead_ends_raw.md, E1-E3)
# --------------------------------------------------------------------------

#: E1 reads a chain's sections this far apart (um), at 1, 3, 5 ... um from
#: its tip, and its tip's stretch is its first this many microns: five
#: sections, the ones a ring is fitted at.
RING_SECTION_STEP_UM = 2.0
RING_TIP_REACH_UM = 10.0
#: A fitted ring is the chain's own when its centre lies within this, or
#: this share of its diameter, of the centreline: the ring fit's own
#: centring limit (``haemodynamics.endothelial._gate``'s ``off_centre``).
RING_CENTRE_LIMIT_UM = 1.5
RING_CENTRE_LIMIT_FRACTION = 0.3
#: E2 judges each point in a window reaching this many of its lumen radii
#: round it, and at least its radius plus this many microns: enough to
#: hold its own ring. A caller asking whether it lies in a wider vessel's
#: lumen passes that vessel's reach.
INSIDE_WALLS_REACH_RADII = 3.0
INSIDE_WALLS_MIN_REACH_UM = 4.0
#: E3 reads the line ahead of a tip every this many microns, from one lumen
#: radius past the tip (its own lumen does not count, as for the mask's
#: ``dead_ends.mask_run_um``) up to this far.
AHEAD_STEP_UM = 1.0
AHEAD_LIMIT_UM = 30.0
#: How the line ahead stops (:attr:`WallsAhead.stop`): in another vessel's
#: lumen with the walls running on all the way, at the image face with them
#: running on, still running on at the limit, or ending in the tissue.
AHEAD_MEETS_VESSEL = "vessel"
AHEAD_FACE = "face"
AHEAD_RUNS_ON = "runs_on"
AHEAD_ENDS = "ends"
#: Whose a fitted ring is (:attr:`RingReading.whose`).
RING_OWN = "own"
RING_OTHER = "other"
RING_OFF_CENTRE = "off_centre"


@dataclass(frozen=True)
class RingReading:
    """One of :meth:`RawEvidenceEndothelial.ring_along`'s ring fits."""

    #: Where along the chain, microns from its tip.
    at_um: float
    #: ``"ring"`` when a wall ring was seen round the section, else why not:
    #: ``fit_ring_section``'s reason (``no_dark_lumen``, ``ring_not_closed``,
    #: ``off_volume``, ``fit_failed``), ``low_contrast`` or ``partial_wall``.
    status: str
    #: The fitted ring's centre (physical ``(z, y, x)`` microns), internal
    #: diameter and contrast, whenever the section was fitted.
    centre_um: Optional[tuple] = None
    diameter_um: Optional[float] = None
    contrast: Optional[float] = None
    #: A seen ring's owner: :data:`RING_OWN`, :data:`RING_OTHER` (its centre
    #: nearer another centreline than the chain) or :data:`RING_OFF_CENTRE`.
    whose: Optional[str] = None


@dataclass(frozen=True)
class RingSupport:
    """E1, :meth:`RawEvidenceEndothelial.ring_along`'s answer for one chain:
    is there a wall ring round it, and is it the chain's own?"""

    #: The share of the chain's sections, every :data:`RING_SECTION_STEP_UM`,
    #: with a wall on both sides of the centreline across it
    #: (:meth:`RawEvidence.vesselness`), over the whole chain and over its
    #: tip's stretch (:data:`RING_TIP_REACH_UM`).
    fraction: float
    tip_fraction: float
    sections: int
    tip_sections: int
    #: The ring fitted at each of the tip's sections; empty when not asked for.
    readings: tuple = ()

    @property
    def rings_seen(self) -> int:
        return sum(1 for r in self.readings if r.status == "ring")

    def rings(self, whose: str) -> int:
        """How many seen rings are *whose* (:data:`RING_OWN` ...)."""
        return sum(1 for r in self.readings if r.status == "ring" and r.whose == whose)

    @property
    def own_ring_fraction(self) -> Optional[float]:
        """The share of the fitted sections whose ring is the chain's own;
        None when none was fitted."""
        return self.rings(RING_OWN) / len(self.readings) if self.readings else None


@dataclass(frozen=True, eq=False)
class InsideWalls:
    """E2, :meth:`RawEvidenceEndothelial.inside_walls`' answer, per point."""

    #: Whether each point lies in a lumen the walls enclose, off the walls.
    lumen: np.ndarray
    #: Whether each point lies on a wall (a capillary under the blur reads
    #: as one bright line, so its centreline is on "wall").
    wall: np.ndarray
    #: Whether each point's lumen -- the enclosed region it lies in, walls
    #: between regions -- also holds a point of another centreline.
    shared: np.ndarray

    @property
    def lumen_fraction(self) -> float:
        return float(self.lumen.mean()) if len(self.lumen) else 0.0

    @property
    def shared_fraction(self) -> float:
        return float(self.shared.mean()) if len(self.shared) else 0.0


@dataclass(frozen=True)
class WallsAhead:
    """E3, :meth:`RawEvidenceEndothelial.walls_ahead`'s answer for one tip:
    does the vessel go on past it along its heading, and into what?"""

    #: How far the vessel reads on, without a break, from :attr:`start_um`
    #: past the tip (stopping where it meets another vessel's lumen).
    run_um: float
    #: :data:`AHEAD_MEETS_VESSEL`, :data:`AHEAD_FACE`, :data:`AHEAD_RUNS_ON`
    #: or :data:`AHEAD_ENDS`.
    stop: str
    #: How far ahead of the tip the line enters another vessel's lumen; None
    #: when it does not within the limit.
    meets_um: Optional[float]
    #: How far ahead of the tip the line leaves the image; None when the face
    #: is past the limit.
    face_um: Optional[float]
    #: The stretches that do not read as vessel before the line meets another
    #: vessel (or within the limit): how many, their length, the longest (um).
    gap_stretches: int
    gap_um: float
    longest_gap_um: float
    #: Where the readings start past the tip and how far apart they are
    #: (um), and each one's vesselness (0 to 1), for drawing.
    start_um: float
    step_um: float
    profile: tuple = ()


#: E4 follows the walls a step at a time, turning at most this far off its
#: last direction at a step (candidates at half and all of it, each all
#: round), and never more than this far off the tip's heading: the dead-end
#: classifier's own "ahead" (``graph.dead_ends.AHEAD_MAX_TURN_DEG``, the
#: facing join's turn limit). At 100 deg it turned along the image's face and
#: into neighbours on E14.5.
ROUTE_CONE_DEG = 30.0
ROUTE_MAX_TURN_DEG = 60.0
#: Each step the route is re-centred between the walls within this, across
#: its direction (um).
ROUTE_RECENTRE_UM = 0.5
#: A stretch that does not read as vessel is crossed, straight on, up to this
#: long (a speck, a fold of the blur); a longer one ends the route.
ROUTE_MAX_GAP_UM = 2.0
#: Each direction is scored over this many steps along it, so the route
#: carries on through a short gap in the wall rather than turning along the
#: gap's edge, and turns where the vessel does.
ROUTE_LOOKAHEAD_STEPS = 3


@dataclass(frozen=True)
class RouteAhead:
    """E4, :meth:`RawEvidenceEndothelial.route_ahead`'s answer for one tip:
    where the walls lead past it, turning where the vessel turns."""

    #: The route, physical ``(z, y, x)`` microns: the tip, then each step from
    #: one lumen radius past it, up to the last that read as vessel.
    path_um: tuple
    #: How far the route followed walls, from :attr:`start_um` past the tip to
    #: its last reading as vessel, gaps it crossed included (um).
    run_um: float
    #: :data:`AHEAD_MEETS_VESSEL`, :data:`AHEAD_FACE`, :data:`AHEAD_RUNS_ON`
    #: or :data:`AHEAD_ENDS`, as for :class:`WallsAhead`.
    stop: str
    #: The straight distance from the tip to the route's end, and the angle
    #: between the tip's heading and that chord; the largest such angle to
    #: any point the route read as vessel (degrees). Every step's direction
    #: is within :data:`ROUTE_MAX_TURN_DEG`; re-centring across it can carry
    #: the chord a little further.
    reach_um: float
    turn_deg: float
    bend_deg: float
    #: How far along the route it entered another vessel's lumen; None when
    #: it did not.
    meets_um: Optional[float]
    #: The dark stretches it crossed: how many, their length, the longest (um).
    gap_stretches: int
    gap_um: float
    longest_gap_um: float
    start_um: float
    step_um: float
    #: Each step's vesselness (0 to 1), for drawing.
    profile: tuple = ()


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
    as given, or with ``auto`` as ``raw_stain.read_raw_stain`` tells it -- the
    one rule, so the build and the Diameters stage never name the same image
    differently; the reading is kept on the calibration. None, with a log
    line saying why, when the image has nothing to read: the stain cannot be
    told, no block with lumen, wall and background in it, or the level the
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
    reading: Optional[dict] = None
    if stain == AUTO:
        from .raw_stain import read_raw_stain

        told = read_raw_stain(raw, mask, voxel_size_zyx, seed=seed)
        if told.stain is None:
            logger.info(
                "Raw evidence: the raw image %s, so the stain cannot be told; set raw_image_stain. "
                "Not read.",
                told.describe(),
            )
            return None
        stain, reading = told.stain, told.as_dict()
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
    calibration = RawCalibration(
        stain=stain, vessel=core_level, wall=wall_level, background=background_level,
        background_spread=spread, contrast=contrast, blocks=used, sigma_vox=float(sigma_vox),
        reading=reading,
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

    def _read(self, lo: np.ndarray, hi: np.ndarray, *, blur: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """The blurred window ``[lo, hi)`` (voxel indices, clipped to the
        image), padded by the blur's reach so it has no edge, as float32,
        with the index its corner sits at. Unblurred, and unpadded, when
        *blur* is False: a ring fit models the image's blur itself."""
        sigma = self.sigma_vox if blur else 0.0
        pad = int(np.ceil(3.0 * sigma)) if sigma > 0 else 0
        plo = np.maximum(np.asarray(lo, dtype=int) - pad, 0)
        phi = np.minimum(np.asarray(hi, dtype=int) + pad, np.asarray(self.shape))
        if np.any(phi <= plo):
            return np.zeros((0, 0, 0), dtype=np.float32), plo
        crop = np.asarray(self.raw[tuple(slice(int(a), int(b)) for a, b in zip(plo, phi))], dtype=np.float32)
        self.stats.queries += 1
        self.stats.voxels_read += int(crop.size)
        if sigma > 0 and crop.size:
            from scipy.ndimage import gaussian_filter

            crop = gaussian_filter(crop, sigma)
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
        return self._normalise(self._wall_pairs(points, radius_um, tangents).max(axis=1))

    def _wall_pairs(self, points: np.ndarray, radius_um: float, tangents: np.ndarray) -> np.ndarray:
        """The raw value of the walls on both sides of each point across its
        tangent, one per pair of opposite directions round it: their best is
        :meth:`vesselness` (before it is normalised); their mean says how
        much of the ring is wall, which only a tangent along the vessel
        satisfies."""
        # Rings across the vessel round each point, at its own radius and a
        # wall's reach either side. A vessel running through the point has
        # wall on *both* sides of it, so the reading is the best pair of
        # opposite directions, each the brightest of its radii: the dimmer of
        # the two. The brightest single point would read any wall within
        # reach -- in a capillary bed, nearly always one. Read across, never
        # along, so a gap in the wall along the path is not filled by the
        # wall either side of it.
        p, q = _across(tangents)
        angles = np.linspace(0.0, 2.0 * np.pi, _RING_DIRECTIONS, endpoint=False)
        offsets = []
        for extra in _RING_RADII_UM:
            r = max(float(radius_um) + extra, 0.5 * float(self._spacing.min()))
            for angle in angles:
                offsets.append(r * (np.cos(angle) * p + np.sin(angle) * q))
        ring = points[:, None, :] + np.stack(offsets, axis=1)  # (points, radii x directions, 3)
        values = self._values(ring.reshape(-1, 3)).reshape(len(points), len(_RING_RADII_UM), _RING_DIRECTIONS)
        by_direction = values.max(axis=1)
        half = _RING_DIRECTIONS // 2
        return np.minimum(by_direction[:, :half], by_direction[:, half:])

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

    # -- dead ends (PLAN_dead_ends_raw.md) --------------------------------

    def ring_along(
        self,
        path_um: np.ndarray,
        radius_um: float,
        *,
        others_um: Optional[np.ndarray] = None,
        fit_rings: bool = True,
        min_contrast: Optional[float] = None,
        psf_sigma_zyx: Optional[Sequence[float]] = None,
        wall_um: Optional[float] = None,
        average_um: Optional[float] = None,
    ) -> RingSupport:
        """E1: whether a wall ring runs round the chain *path_um* (tip first),
        of lumen radius about *radius_um*.

        Every :data:`RING_SECTION_STEP_UM` along it, a wall on both sides of
        the centreline across it (:meth:`vesselness`): the share of sections
        that have one, over the chain and over its tip's stretch. At the tip's
        sections a ring is also fitted
        (``haemodynamics.endothelial.fit_ring_section``, on a window round
        the section) when *fit_rings*, to say whose it is: the chain's own,
        centred on it, or another centreline's (*others_um*, the points of
        the centrelines near it that are not the chain) -- a spur inside a
        fat lumen sits inside its parent's ring and has none of its own.
        A ring is seen when its contrast reaches *min_contrast*
        (``endothelial_min_ring_contrast``'s default when None) and its wall
        runs round it. The blur and the wall are *psf_sigma_zyx* and
        *wall_um*, else the calibration's ring fields, else fitted free (as
        noisy as a section alone)."""
        from haemolynx.haemodynamics.edt_diameter import _centreline_tangents
        from haemolynx.haemodynamics.endothelial import DEFAULT_AVERAGE_ALONG_VESSEL_UM, ENDOTHELIAL_MIN_RING_CONTRAST
        from haemolynx.haemodynamics.fwhm_diameter import _arc_length_parameterize, _interpolate_centerline

        path = _distinct(path_um)
        if len(path) < 2:
            return RingSupport(0.0, 0.0, 0, 0)
        s, total = _arc_length_parameterize(path)
        at = np.arange(0.5 * RING_SECTION_STEP_UM, total, RING_SECTION_STEP_UM)
        if not len(at):
            at = np.array([0.5 * total])
        points = _interpolate_centerline(path, s, at)
        tangents = _centreline_tangents(path, s, total, at)
        walled = self.vesselness(points, radius_um, tangents) >= BRIGHT_THRESHOLD
        tip = at < RING_TIP_REACH_UM
        readings: list[RingReading] = []
        if fit_rings:
            psf = psf_sigma_zyx if psf_sigma_zyx is not None else self.calibration.ring_psf_sigma_zyx
            wall = wall_um if wall_um is not None else self.calibration.ring_wall_um
            gate = ENDOTHELIAL_MIN_RING_CONTRAST if min_contrast is None else float(min_contrast)
            others = None if others_um is None else np.asarray(others_um, dtype=float).reshape(-1, 3)
            for where in at[tip]:
                readings.append(self._ring_reading(
                    path, s, total, float(where), radius_um, psf=psf, wall_um=wall,
                    average_um=DEFAULT_AVERAGE_ALONG_VESSEL_UM if average_um is None else float(average_um),
                    min_contrast=gate, others=others,
                ))
        return RingSupport(
            fraction=float(walled.mean()),
            tip_fraction=float(walled[tip].mean()),
            sections=int(len(at)),
            tip_sections=int(tip.sum()),
            readings=tuple(readings),
        )

    def _ring_reading(
        self, path, s, total, at, radius_um, *, psf, wall_um, average_um, min_contrast, others,
    ) -> RingReading:
        """One section's ring, fitted on a window holding every point the fit
        samples, so it reads as it would on the whole volume."""
        from haemolynx.haemodynamics.edt_diameter import _centreline_tangents
        from haemolynx.haemodynamics.endothelial import MIN_WALL_COVERAGE, fit_ring_section
        from haemolynx.haemodynamics.fwhm_diameter import _interpolate_centerline
        from haemolynx.haemodynamics.sections import section_axes

        started = time.perf_counter()
        guide = 2.0 * max(float(radius_um), 0.5 * float(self._spacing.min()))
        # fit_ring_section's own plane: half-width at most this (its blur
        # is the PSF projected onto the section, or 1 um fitted free),
        # averaged over average_um of the centreline.
        blur = float(max(psf)) if psf is not None else 1.0
        extent = max(6.0, 1.2 * guide + 3.0 + 3.0 * blur)
        half = 0.5 * max(float(average_um), 0.0)
        lo_s, hi_s = max(0.0, at - half), min(total, at + half)
        spine = np.vstack([path[(s >= lo_s) & (s <= hi_s)], _interpolate_centerline(path, s, np.array([lo_s, at, hi_s]))])
        reach = np.sqrt(2.0) * (extent + 1.0) + 2.0 * self._spacing
        lo = np.maximum(np.floor((spine.min(axis=0) - reach) / self._spacing).astype(int), 0)
        hi = np.minimum(np.ceil((spine.max(axis=0) + reach) / self._spacing).astype(int) + 1, np.asarray(self.shape))
        if np.any(hi <= lo):
            return RingReading(at, "off_volume")
        window, corner = self._read(lo, hi, blur=False)
        fit, why = fit_ring_section(
            window, path - corner * self._spacing, s, total, at, self.voxel_size_zyx, guide_um=guide,
            psf_sigma_zyx=psf, average_um=average_um, wall_um=wall_um,
        )
        self.stats.seconds += time.perf_counter() - started
        if fit is None:
            return RingReading(at, why)
        _t, a, b = section_axes(_centreline_tangents(path, s, total, np.array([at]))[0])
        point = _interpolate_centerline(path, s, np.array([at]))[0]
        centre = point + fit.centre_ab_um[0] * a + fit.centre_ab_um[1] * b
        status = "ring"
        if fit.contrast < min_contrast:
            status = "low_contrast"
        elif fit.wall_coverage < MIN_WALL_COVERAGE:
            status = "partial_wall"
        whose = None
        if status == "ring":
            # The ring is the nearest centreline's: a wide ring's centring
            # limit is generous, and would lend a neighbour's ring to a
            # strand running inside it.
            if others is not None and len(others) and (
                float(np.linalg.norm(others - centre, axis=1).min())
                < float(np.linalg.norm(path - centre, axis=1).min())
            ):
                whose = RING_OTHER
            elif fit.centre_offset_um <= max(RING_CENTRE_LIMIT_UM, RING_CENTRE_LIMIT_FRACTION * fit.diameter_um):
                whose = RING_OWN
            else:
                whose = RING_OFF_CENTRE
        return RingReading(
            at, status, centre_um=tuple(float(v) for v in centre), diameter_um=float(fit.diameter_um),
            contrast=float(fit.contrast), whose=whose,
        )

    def inside_walls(
        self,
        points_um: np.ndarray,
        radius_um: float,
        *,
        others_um: Optional[np.ndarray] = None,
        reach_um: Optional[float] = None,
    ) -> InsideWalls:
        """E2: whether walls enclose each point, and whether the lumen they
        enclose there is shared with another centreline (*others_um*) -- a
        chain running inside another vessel's walls is two centrelines in one
        lumen, seen in the raw image instead of the mask.

        Each point is judged on its own window, reaching *reach_um* round it
        (:data:`INSIDE_WALLS_REACH_RADII` of *radius_um*, at least the
        radius plus :data:`INSIDE_WALLS_MIN_REACH_UM`, when None), blurred
        and normalised as every reading is: walls are what reads as the stain
        (:data:`BRIGHT_THRESHOLD`), and a lumen is what they enclose in some
        slice (``segmentation_raw_comparison.enclosed_by_walls``, once per
        window). The answer at a point never depends on the others asked
        with it. Near the chain's own junction its lumen opens into its
        parent's, so read the share along the chain, not one point."""
        from scipy import ndimage

        from .segmentation_raw_comparison import enclosed_by_walls

        started = time.perf_counter()
        points = np.asarray(points_um, dtype=float).reshape(-1, 3)
        count = len(points)
        lumen, wall, shared = (np.zeros(count, dtype=bool) for _ in range(3))
        radius = max(float(radius_um), 0.0)
        reach = float(reach_um) if reach_um is not None else max(
            INSIDE_WALLS_REACH_RADII * radius, radius + INSIDE_WALLS_MIN_REACH_UM
        )
        half = np.ceil(reach / self._spacing).astype(int)
        shape = np.asarray(self.shape)
        others = (
            np.rint(np.asarray(others_um, dtype=float).reshape(-1, 3) / self._spacing).astype(int)
            if others_um is not None else np.zeros((0, 3), dtype=int)
        )
        for i, point in enumerate(points):
            centre = np.rint(point / self._spacing).astype(int)
            if np.any(centre < 0) or np.any(centre >= shape):
                continue
            lo, hi = np.maximum(centre - half, 0), np.minimum(centre + half + 1, shape)
            crop, corner = self._read(lo, hi)
            start = lo - corner
            window = crop[tuple(slice(int(a), int(a + n)) for a, n in zip(start, hi - lo))]
            walls = self._normalise(window) >= BRIGHT_THRESHOLD
            here = tuple(int(v) for v in centre - lo)
            if walls[here]:
                wall[i] = True
                continue
            open_lumen = enclosed_by_walls(walls) & ~walls
            if not open_lumen[here]:
                continue
            lumen[i] = True
            if len(others):
                local = others - lo
                within = np.all((local >= 0) & (local < hi - lo), axis=1)
                if within.any():
                    labels, _count = ndimage.label(open_lumen)
                    hits = local[within]
                    shared[i] = bool(np.any(labels[hits[:, 0], hits[:, 1], hits[:, 2]] == labels[here]))
        self.stats.seconds += time.perf_counter() - started
        return InsideWalls(lumen=lumen, wall=wall, shared=shared)

    def walls_ahead(
        self,
        tip_um: Sequence[float],
        heading: Sequence[float],
        radius_um: float,
        *,
        others_um: Optional[np.ndarray] = None,
        others_radius_um: Any = 0.0,
        limit_um: float = AHEAD_LIMIT_UM,
        step_um: float = AHEAD_STEP_UM,
    ) -> WallsAhead:
        """E3: whether the vessel goes on past a tip along its *heading* (a
        unit vector out of the tip, ``dead_ends.tip_heading``), the raw
        analogue of ``dead_ends.mask_run_um``.

        From one lumen radius past the tip, every *step_um* up to *limit_um*
        further, the line reads as vessel where a wall runs on both sides of
        it across the heading (:meth:`vesselness`). It stops where it enters
        another vessel's lumen -- within *others_radius_um* (one value, or
        one per point) of a point of *others_um*, the centrelines that are
        not the chain -- or leaves the image. :attr:`WallsAhead.stop` says
        how it ended, and the gaps before it how much of the way was not
        vessel: a join across a gap would need the walls to run on across
        it (Q2), a vessel whose plasma faded before the face runs on to it
        (Q3). Straight along the heading: a vessel that turns away soon
        after the tip reads as ending."""
        tip = np.asarray(tip_um, dtype=float).reshape(3)
        direction = np.asarray(heading, dtype=float).reshape(3)
        norm = float(np.linalg.norm(direction))
        if not norm > 0:
            raise ValueError("walls_ahead needs a heading of non-zero length")
        direction = direction / norm
        start = max(float(radius_um), 0.0)
        step = float(step_um)
        offsets = start + step * np.arange(1, int(np.ceil(float(limit_um) / step)) + 1)
        samples = tip + offsets[:, None] * direction
        extent = (np.asarray(self.shape, dtype=float) - 1.0) * self._spacing
        in_image = np.all((samples >= -1e-6) & (samples <= extent + 1e-6), axis=1)
        leaving = int(np.argmin(in_image)) if not in_image.all() else None
        face_um = None
        if leaving is not None:
            samples, offsets = samples[:leaving], offsets[:leaving]
            face_um = float(_distance_along_to_box(tip, direction, extent))
        readings = (
            self.vesselness(samples, radius_um, np.tile(direction, (len(samples), 1)))
            if len(samples) else np.zeros(0)
        )
        vessel = readings >= BRIGHT_THRESHOLD
        meets = None
        if others_um is not None and len(samples):
            others = np.asarray(others_um, dtype=float).reshape(-1, 3)
            if len(others):
                from scipy.spatial import cKDTree

                distance, nearest = cKDTree(others).query(samples)
                limit = np.broadcast_to(np.asarray(others_radius_um, dtype=float), (len(others),))[nearest]
                entered = np.flatnonzero(distance <= limit)
                meets = int(entered[0]) if len(entered) else None
        judged = vessel[:meets] if meets is not None else vessel
        run = int(np.argmin(judged)) if not judged.all() else len(judged)
        dark = ~judged
        stretches = int(np.count_nonzero(dark[1:] & ~dark[:-1]) + (1 if len(dark) and dark[0] else 0))
        if meets is not None and judged.all():
            stop = AHEAD_MEETS_VESSEL
        elif meets is None and vessel.all():
            stop = AHEAD_FACE if leaving is not None else AHEAD_RUNS_ON
        else:
            stop = AHEAD_ENDS
        return WallsAhead(
            run_um=float(run * step),
            stop=stop,
            meets_um=None if meets is None else float(offsets[meets]),
            face_um=face_um,
            gap_stretches=stretches,
            gap_um=float(np.count_nonzero(dark) * step),
            longest_gap_um=float(_longest_run(dark) * step),
            start_um=start,
            step_um=step,
            profile=tuple(float(v) for v in readings),
        )

    def route_ahead(
        self,
        tip_um: Sequence[float],
        heading: Sequence[float],
        radius_um: float,
        *,
        others_um: Optional[np.ndarray] = None,
        others_radius_um: Any = 0.0,
        limit_um: float = AHEAD_LIMIT_UM,
        step_um: float = AHEAD_STEP_UM,
        cone_deg: float = ROUTE_CONE_DEG,
        max_turn_deg: float = ROUTE_MAX_TURN_DEG,
        max_gap_um: float = ROUTE_MAX_GAP_UM,
    ) -> RouteAhead:
        """E4: where the walls lead past a tip, followed round the vessel's
        turns -- :meth:`walls_ahead` along a route instead of a straight line.

        From one lumen radius past the tip, a step of *step_um* at a time up to
        *limit_um*: of the directions within *cone_deg* of the last (straight
        on first, so it wins a tie), and within *max_turn_deg* of the tip's
        *heading*, the route takes the one with the most of the ring round it
        wall (unclipped, so walls brighter than the run's level still rank:
        :meth:`vesselness`'s best pair would be met by a direction across the
        vessel too), and re-centres within :data:`ROUTE_RECENTRE_UM`. A step
        reads as vessel by :meth:`vesselness`, as :meth:`walls_ahead`'s
        readings do. Where no direction reads as vessel it
        carries straight on, and a dark stretch longer than *max_gap_um* ends
        it. It stops on entering another vessel's lumen (*others_um*,
        *others_radius_um*, as for :meth:`walls_ahead`) or leaving the image.

        A tracker rather than a least-cost route: with no end point given, a
        route must choose one, and choosing the nearest centreline is what
        made grid G5's route through the mask join neighbours (no better than
        never joining, 23/29). With *cone_deg* 0 (no turn, no re-centring) and
        *max_gap_um* 0 it reads :meth:`walls_ahead`'s line and stops where it
        does."""
        tip = np.asarray(tip_um, dtype=float).reshape(3)
        first = np.asarray(heading, dtype=float).reshape(3)
        norm = float(np.linalg.norm(first))
        if not norm > 0:
            raise ValueError("route_ahead needs a heading of non-zero length")
        first = first / norm
        start = max(float(radius_um), 0.0)
        step = float(step_um)
        extent = (np.asarray(self.shape, dtype=float) - 1.0) * self._spacing
        tree = limits = None
        if others_um is not None:
            others = np.asarray(others_um, dtype=float).reshape(-1, 3)
            if len(others):
                from scipy.spatial import cKDTree

                tree = cKDTree(others)
                limits = np.broadcast_to(np.asarray(others_radius_um, dtype=float), (len(others),))
        least_cos = float(np.cos(np.radians(max_turn_deg)))
        point, direction = tip + start * first, first
        path, profile, gaps = [tip.copy()], [], []
        travelled = run = gap = bend = 0.0
        last_vessel = 1
        meets, stop = None, None
        for _ in range(int(np.ceil(float(limit_um) / step))):
            candidates = _cone(direction, cone_deg)
            candidates = candidates[candidates @ first >= least_cos - 1e-9]
            ahead = point + step * candidates
            inside = np.all((ahead >= -1e-6) & (ahead <= extent + 1e-6), axis=1)
            # A route that may turn stops once its own direction leaves the
            # image within the look-ahead: turning along the face would follow
            # the reading's own edge (ring samples past it read as background).
            straight_on = point + ROUTE_LOOKAHEAD_STEPS * step * direction
            heading_out = cone_deg > 0 and (np.any(straight_on < -1e-6) or np.any(straight_on > extent + 1e-6))
            if not inside.any() or heading_out:
                # The image ends ahead: the walls ran on to its face, or not.
                stop = AHEAD_FACE if gap == 0.0 else AHEAD_ENDS
                break
            candidates, ahead = candidates[inside], ahead[inside]
            # Steered by the wall all round over the steps ahead, judged at
            # the first as walls_ahead judges.
            steps = np.arange(1, ROUTE_LOOKAHEAD_STEPS + 1) * step
            along = (point + steps[:, None, None] * candidates[None]).reshape(-1, 3)
            pairs = self._lift(self._wall_pairs(along, radius_um, np.tile(candidates, (len(steps), 1))))
            pairs = pairs.reshape(len(steps), len(candidates), -1)
            best = int(np.argmax(pairs.mean(axis=(0, 2))))
            lift = pairs[0].max(axis=1)
            vessel = lift[best] >= BRIGHT_THRESHOLD
            if vessel:
                direction = candidates[best]
                point = self._recentre(ahead[best], direction, radius_um) if cone_deg > 0 else ahead[best]
            else:
                point = point + step * direction
            travelled += step
            if not vessel and (np.any(point < -1e-6) or np.any(point > extent + 1e-6)):
                # Carried straight on, through a gap, out of the image.
                stop = AHEAD_ENDS
                break
            path.append(point.copy())
            profile.append(float(np.clip(lift[best], 0.0, 1.0)))
            if vessel:
                if gap > 0.0:
                    gaps.append(gap)
                    gap = 0.0
                last_vessel, run = len(path), travelled
                chord = point - tip
                length = float(np.linalg.norm(chord))
                if length > 0:
                    bend = max(bend, float(np.degrees(np.arccos(np.clip(chord @ first / length, -1.0, 1.0)))))
            else:
                gap += step
                if gap > float(max_gap_um) + 1e-9:
                    stop = AHEAD_ENDS
                    break
            if tree is not None:
                distance, nearest = tree.query(point)
                if distance <= limits[nearest]:
                    meets, stop = travelled, AHEAD_MEETS_VESSEL
                    last_vessel, run = len(path), travelled
                    break
        if stop is None:
            stop = AHEAD_RUNS_ON if gap == 0.0 else AHEAD_ENDS
        path = path[:last_vessel]
        chord = path[-1] - tip
        reach = float(np.linalg.norm(chord))
        turn = float(np.degrees(np.arccos(np.clip(chord @ first / reach, -1.0, 1.0)))) if reach > 0 else 0.0
        return RouteAhead(
            path_um=tuple(tuple(float(v) for v in p) for p in path),
            run_um=float(run),
            stop=stop,
            reach_um=reach,
            turn_deg=turn,
            bend_deg=bend,
            meets_um=meets,
            gap_stretches=len(gaps),
            gap_um=float(sum(gaps)),
            longest_gap_um=float(max(gaps, default=0.0)),
            start_um=start,
            step_um=step,
            profile=tuple(profile),
        )

    def _lift(self, values: np.ndarray) -> np.ndarray:
        """Raw values above the background in units of the stain's level,
        unclipped: walls brighter than the run's level still rank."""
        return (np.asarray(values, dtype=float) - self.calibration.background) / self._scale

    def _recentre(self, point: np.ndarray, direction: np.ndarray, radius_um: float) -> np.ndarray:
        """*point* moved across *direction*, within :data:`ROUTE_RECENTRE_UM`,
        to where the most of the ring round it is wall (staying put on a tie)."""
        p, q = _across(direction[None])
        r = ROUTE_RECENTRE_UM
        offsets = [(0.0, 0.0)] + [(a, b) for a in (-r, 0.0, r) for b in (-r, 0.0, r) if (a, b) != (0.0, 0.0)]
        candidates = point + np.asarray([a * p[0] + b * q[0] for a, b in offsets])
        pairs = self._wall_pairs(candidates, radius_um, np.tile(direction, (len(offsets), 1)))
        return candidates[int(np.argmax(pairs.mean(axis=1)))]


def _cone(direction: np.ndarray, cone_deg: float, azimuths: int = 8) -> np.ndarray:
    """*direction* first, then unit directions at half and all of *cone_deg*
    off it, *azimuths* round each."""
    direction = np.asarray(direction, dtype=float).reshape(3)
    if cone_deg <= 0:
        return direction[None]
    p, q = _across(direction[None])
    out = [direction]
    for angle in np.radians([0.5 * cone_deg, cone_deg]):
        for phi in np.linspace(0.0, 2.0 * np.pi, azimuths, endpoint=False):
            out.append(np.cos(angle) * direction + np.sin(angle) * (np.cos(phi) * p[0] + np.sin(phi) * q[0]))
    return np.asarray(out)


def _distinct(path_um) -> np.ndarray:
    """*path_um* as an ``(n, 3)`` array without repeated consecutive points,
    so its arc length rises strictly."""
    path = np.asarray(path_um, dtype=float).reshape(-1, 3)
    if len(path) < 2:
        return path
    keep = np.concatenate([[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9])
    return path[keep]


def _distance_along_to_box(point: np.ndarray, direction: np.ndarray, extent: np.ndarray) -> float:
    """How far from *point* along unit *direction* the box ``[0, extent]`` ends."""
    limits = []
    for axis in range(3):
        if direction[axis] > 1e-8:
            limits.append((extent[axis] - point[axis]) / direction[axis])
        elif direction[axis] < -1e-8:
            limits.append(point[axis] / -direction[axis])
    return float(min(limits)) if limits else float("inf")


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


def ring_calibration(
    G: Any,
    evidence: RawEvidence,
    *,
    guide_attribute: Optional[str] = "edt_diameter_um",
    min_contrast: Optional[float] = None,
    seed: int = 0,
    edges: Any = None,
    workers: int = 1,
    memmap_directory: Any = None,
) -> RawCalibration:
    """*evidence*'s calibration with its ring fields filled from the image's
    own wide vessels' rings (``endothelial.calibrate_from_rings``, its
    arguments): the blur, the wall's thickness, and the median fitted wall
    brightness of the rings that pass the contrast gate. Unchanged, with the
    calibration's log line saying why, when the rings do not calibrate.

    Optional and costly (90-120 s on E14.5 with 8 workers, against the block
    calibration's second): a run that has measured endothelial diameters has
    paid for it already. Assign the result to ``evidence.calibration`` for
    :meth:`RawEvidenceEndothelial.ring_along` to fit with it."""
    from haemolynx.haemodynamics import endothelial

    gate = endothelial.ENDOTHELIAL_MIN_RING_CONTRAST if min_contrast is None else float(min_contrast)
    fits = endothelial.ring_calibration_fits(
        G, evidence.raw, evidence.voxel_size_zyx, guide_attribute=guide_attribute, seed=seed, edges=edges,
        workers=workers, memmap_directory=memmap_directory,
    )
    psf, wall_um, _details = endothelial.calibrate_from_rings(
        G, evidence.raw, evidence.voxel_size_zyx, guide_attribute=guide_attribute, min_contrast=gate,
        seed=seed, edges=edges, fits=fits,
    )
    if psf is None or wall_um is None:
        return evidence.calibration
    levels = [
        fit.wall_level
        for readings in fits.readings for fit, _why in readings
        if fit is not None and endothelial._gate(fit, min_contrast=gate, min_diameter_um=0.0) == "ok"
    ]
    return evidence.calibration.with_rings(psf, wall_um, float(np.median(levels)) if levels else None)


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
    """The raw image a run's settings name: the one raw image of the Input tab
    (``raw_image_path`` / ``raw_image_channel`` / ``raw_image_stain``). The
    names it replaced (``fwhm_raw_tiff_path``, ``endothelial_image_path``)
    are read into it when a config loads (``Schema.upgrade``), so they are
    not looked for here. None when the run names no raw image."""
    path = settings.get("raw_image_path")
    if path in (None, ""):
        return None
    channel = settings.get("raw_image_channel")
    named = settings.get("raw_image_stain")
    stain = str(named) if named in STAINS else AUTO
    return RawImageSpec(Path(path), None if channel in (None, "") else int(channel), stain, "raw_image_path")


def resident_bytes(volume: Any) -> int:
    """The bytes of RAM *volume* keeps: the whole array a view was cut from
    (a crop, a transpose, one channel of a composite read whole), and 0 for
    a memory map, whose pages are the operating system's to drop."""
    array = volume
    while isinstance(array, np.ndarray) and not isinstance(array, np.memmap) and isinstance(array.base, np.ndarray):
        array = array.base
    if not isinstance(array, np.ndarray) or isinstance(array, np.memmap):
        return 0
    return int(array.nbytes)


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
    from haemolynx.haemodynamics.fwhm_diameter import _read_tiff_channel, load_single_channel_tiff_volume
    from haemolynx.io.axis_order import apply_axis_order, axis_order_for_file

    if use_memmap:
        return load_single_channel_tiff_volume(
            path, axis_order=axis_order, use_memmap=True, memmap_directory=memmap_directory, channel=channel,
        )
    return apply_axis_order(
        _read_tiff_channel(Path(path), 0 if channel is None else int(channel)),
        axis_order_for_file(axis_order, path),
    )


def _raw_downsample_factor(settings: Mapping[str, Any]) -> int:
    """``raw_image_downsample``'s factor, 1 when the settings do not say."""
    from haemolynx.io.downsample import RAW_SETTING, downsample_factor

    return downsample_factor(settings.get(RAW_SETTING))


def raw_evidence_from_settings(
    settings: Mapping[str, Any],
    mask: np.ndarray,
    voxel_size_zyx: Sequence[float],
    *,
    box: Any = None,
    downsample: Any = None,
) -> Optional[RawEvidence]:
    """The :class:`RawEvidence` a run's settings give: its raw image loaded
    (:func:`load_raw_channel`), cut to the run's crop *box*, reduced onto the
    run's grid by *downsample* (an ``io.downsample.Downsample``, the segmented
    image's, with ``raw_image_downsample`` read from *settings*), calibrated
    round *mask* (:func:`calibrate_raw`). None when the run names no raw image
    or the image has nothing to read; a misaligned image is refused."""
    spec = raw_image_for_build(settings)
    if spec is None:
        return None
    from haemolynx.io.axis_order import axis_order_for_run
    from haemolynx.io.load import resolve_image_path_with_optional_zip

    try:
        path = resolve_image_path_with_optional_zip(spec.path)
    except FileNotFoundError:
        path = spec.path
    if not path.is_file():
        raise FileNotFoundError(
            f"The raw image ({spec.setting}) is not there: {spec.path}. Fix: point raw_image_path "
            "at the raw image on the Input tab, or clear it."
        )
    started = time.perf_counter()
    raw = load_raw_channel(
        path,
        spec.channel,
        axis_order=axis_order_for_run(settings),
        use_memmap=bool(settings.get("use_memmap_loading")),
        memmap_directory=settings.get("memmap_directory"),
    )
    shape = tuple(int(n) for n in np.shape(raw))
    reduced = downsample is not None and not downsample.is_identity
    if box is not None:
        expected = tuple(int(n) for n in box.source_shape_zyx)
    elif reduced:
        expected = tuple(int(n) for n in downsample.source_shape_zyx)
    else:
        expected = tuple(int(n) for n in mask.shape)
    if shape != expected:
        raise ValueError(
            f"The raw image {spec.path.name} ({spec.setting}) has shape {shape}, not the image's "
            f"{expected}: it must be the same stack the segmentation was made from"
        )
    framed = raw[box.slices] if box is not None and not box.is_whole_volume else raw
    raw_factor = _raw_downsample_factor(settings)
    if reduced or raw_factor > 1:
        from haemolynx.io.downsample import downsample_for, downsample_raw_volume

        if downsample is None:
            # The segmented image was not reduced: the run's grid is the
            # mask's own, which a raw factor alone is read against.
            downsample = downsample_for(1, mask.shape, voxel_size_zyx)
        # Onto the run's grid, a copy the box's size or smaller: the window
        # reads below then come from it, not from the whole image.
        framed = downsample_raw_volume(
            framed, downsample, raw_factor, description="raw image",
            use_memmap=bool(settings.get("use_memmap_loading")),
            memmap_directory=settings.get("memmap_directory"),
        )
        raw, box = framed, None
    calibration = calibrate_raw(framed, mask, voxel_size_zyx, stain=spec.stain)
    if calibration is None:
        return None
    evidence = make_raw_evidence(raw, mask.shape, voxel_size_zyx, calibration, box=box)
    held = resident_bytes(evidence.raw)
    logger.info(
        "Raw evidence from %s%s (%s): %s; loaded and calibrated in %.1f s, %s.",
        spec.path.name,
        "" if spec.channel is None else f" channel {spec.channel}",
        spec.setting,
        calibration.describe(),
        time.perf_counter() - started,
        f"{held / 2**20:.0f} MB of it in RAM" if held else "memory-mapped",
    )
    return evidence


__all__ = [
    "AHEAD_ENDS",
    "AHEAD_FACE",
    "AHEAD_MEETS_VESSEL",
    "AHEAD_RUNS_ON",
    "AUTO",
    "BRIGHT_THRESHOLD",
    "ENDOTHELIAL",
    "GRAPH_ATTRIBUTE",
    "PLASMA_COLUMN",
    "RING_OFF_CENTRE",
    "RING_OTHER",
    "RING_OWN",
    "STAINS",
    "InsideWalls",
    "RawCalibration",
    "RawEvidence",
    "RawEvidenceEndothelial",
    "RawEvidencePlasma",
    "RawEvidenceStats",
    "RawImageSpec",
    "RawSignal",
    "RingReading",
    "RingSupport",
    "RouteAhead",
    "WallsAhead",
    "calibrate_raw",
    "load_raw_channel",
    "make_raw_evidence",
    "raw_evidence_from_settings",
    "raw_image_for_build",
    "resident_bytes",
    "ring_calibration",
]
