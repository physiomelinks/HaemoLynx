"""Is FWHM right? Vessels of known width, planted in the run's own image.

The decoy check (:mod:`.fwhm_decoys`) asks whether FWHM reads a width where
there is no vessel. This asks the other question: where there is one, does it
read the right width? Coverage, fit quality and decoys cannot say; nothing in
the image knows a real vessel's true width.

Beside each of a sample of real vessels, in tissue the segmentation says is
empty, a synthetic vessel is drawn into a copy of the raw image: the real
vessel's centreline moved sideways, a filled lumen of a known width -- the
real vessel's mask width, give or take a quarter, so a measurement guided by
that width does not start at the answer -- blurred by the image's own PSF, as
bright above the local background as the real vessel is, with shot noise of
the image's own scale. Measured with a run's FWHM settings, the planted
vessels say how far off its widths are (:func:`planted_width_report`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from . import fwhm_decoys

__all__ = [
    "DEFAULT_OPTICS_PSF_UM",
    "PLANTED_SAMPLE_SIZE",
    "PlantedVessels",
    "draw_vessel",
    "image_noise_gain",
    "optics_psf",
    "plant_vessels",
    "planted_width_report",
]

#: Vessels planted at most, :data:`PLANTED_PER_VESSEL` beside each sampled
#: vessel until this many.
PLANTED_SAMPLE_SIZE = 60

#: Vessels planted beside each sampled one, each on its own side: one apiece
#: left a five-vessel sample's error swinging with whichever one a setting
#: happened to suit.
PLANTED_PER_VESSEL = 3

#: The optics' blur ``(z, y, x)`` sigma in um when the image's own cannot be
#: estimated: a two-photon objective's, as the diameter benchmark draws with.
DEFAULT_OPTICS_PSF_UM = (1.2, 0.35, 0.35)

#: A planted width is its real vessel's mask width times a factor drawn from
#: 1 +- this, so a measurement guided by the mask width starts off by up to
#: as much as a real one does.
PLANTED_WIDTH_SPREAD = 0.25

#: A planted vessel's centreline keeps at least this many of its own widths,
#: plus twice the blur, from any vessel voxel -- real or planted -- so a
#: transverse profile three widths long reads it alone.
PLANTED_CLEARANCE_MULTIPLE = 1.5

#: Sub-samples per voxel axis drawing a lumen's edge, which also puts the
#: voxels' own box into the blur (see :func:`optics_psf`).
_SUBSAMPLES = 2


@dataclass(frozen=True)
class PlantedVessels:
    """A copy of the raw image with vessels drawn in, and a graph of them."""

    #: The raw image, as float32, with every planted vessel added.
    volume: np.ndarray
    #: One edge per planted vessel, its centreline in ``voxels`` and its true
    #: width in ``planted_diameter_um``, carrying its real neighbour's guide
    #: widths so a measurement guided by them reads it as it would that vessel.
    probe: nx.MultiGraph
    #: The optics' blur they were drawn with.
    optics_psf_um: tuple[float, float, float]
    #: The shot-noise scale they were drawn with (variance per unit of signal).
    noise_gain: float


def optics_psf(
    image_psf_um: Optional[Sequence[float]], voxel_size_zyx: Sequence[float]
) -> tuple[float, float, float]:
    """The optics' share of the blur an image shows: the voxels' own box
    (sigma = voxel / sqrt 12) taken out of *image_psf_um*, which drawing a
    lumen on sub-voxel samples puts back. :data:`DEFAULT_OPTICS_PSF_UM`
    when *image_psf_um* is ``None``."""
    if image_psf_um is None:
        return tuple(float(v) for v in DEFAULT_OPTICS_PSF_UM)
    return tuple(
        float(np.sqrt(max(float(s) ** 2 - float(v) ** 2 / 12.0, (0.1 * float(v)) ** 2)))
        for s, v in zip(image_psf_um, voxel_size_zyx)
    )


def _values_along(volume: np.ndarray, points_um: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """*volume* at the voxels nearest *points_um*, inside it only."""
    index = np.rint(np.asarray(points_um, dtype=float) / spacing).astype(int)
    inside = np.all((index >= 0) & (index < np.asarray(volume.shape)), axis=1)
    index = index[inside]
    return np.asarray(volume[index[:, 0], index[:, 1], index[:, 2]], dtype=float)


def image_noise_gain(volume: np.ndarray, centrelines_um, voxel_size_zyx) -> float:
    """The image's shot-noise scale: the variance of the step from voxel to
    voxel along real vessels' centrelines, halved, over their mean intensity
    -- the median over vessels. Along a vessel the lumen barely changes from
    one voxel to the next, so the steps are mostly noise; for a photon count
    with gain *g* this is *g*. 1.0 when no vessel gives a reading."""
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    gains = []
    for line in centrelines_um:
        values = _values_along(volume, np.asarray(line, dtype=float), spacing)
        if values.size < 5 or float(np.mean(values)) <= 0:
            continue
        gains.append(float(np.var(np.diff(values))) / 2.0 / float(np.mean(values)))
    gains = [g for g in gains if np.isfinite(g) and g > 0]
    return float(np.median(gains)) if gains else 1.0


def _dense(line_um: np.ndarray, step_um: float) -> np.ndarray:
    """*line_um* resampled every *step_um* along its length."""
    line = np.asarray(line_um, dtype=float)
    lengths = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    if lengths[-1] <= 0:
        return line[:1]
    at = np.linspace(0.0, lengths[-1], max(2, int(np.ceil(lengths[-1] / step_um)) + 1))
    return np.stack([np.interp(at, lengths, line[:, axis]) for axis in range(line.shape[1])], axis=1)


def draw_vessel(
    volume: np.ndarray,
    centreline_um,
    diameter_um: float,
    voxel_size_zyx,
    optics_psf_um,
    *,
    peak: float,
    noise_gain: float,
    rng: np.random.Generator,
) -> Optional[tuple[tuple[slice, ...], np.ndarray]]:
    """Add a filled lumen *diameter_um* wide along *centreline_um* to
    *volume*, in place: blurred by *optics_psf_um*, scaled so its centreline
    reads *peak* above what was there, with Gaussian shot noise of variance
    *noise_gain* times the signal, and nothing below zero. Works in a box
    round the vessel only. Returns the box and its lumen (where at least
    half of a voxel is inside), or ``None`` when the vessel misses the
    volume."""
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    optics = np.asarray(optics_psf_um, dtype=float)
    radius = 0.5 * float(diameter_um)
    line = _dense(np.asarray(centreline_um, dtype=float), 0.25 * float(spacing.min()))
    margin = radius + 4.0 * float(optics.max()) + float(spacing.max())
    shape = np.asarray(volume.shape)
    lo = np.maximum(np.floor((line.min(axis=0) - margin) / spacing).astype(int), 0)
    hi = np.minimum(np.ceil((line.max(axis=0) + margin) / spacing).astype(int) + 1, shape)
    if np.any(hi <= lo):
        return None
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    centres = (np.stack(np.indices(tuple(hi - lo)), axis=-1).reshape(-1, 3) + lo) * spacing

    # Only the shell round the lumen's edge needs sub-voxel samples: a voxel
    # whose centre is more than half its diagonal inside is wholly in.
    tree = cKDTree(line)
    half_diagonal = 0.5 * float(np.linalg.norm(spacing))
    distance, _ = tree.query(centres, distance_upper_bound=radius + half_diagonal)
    occupancy = (distance <= radius - half_diagonal).astype(float)
    shell = np.flatnonzero(np.isfinite(distance) & (distance > radius - half_diagonal))
    if shell.size:
        offsets = ((np.arange(_SUBSAMPLES) + 0.5) / _SUBSAMPLES - 0.5)
        inside = np.zeros(shell.size)
        for dz in offsets:
            for dy in offsets:
                for dx in offsets:
                    sub = centres[shell] + np.array([dz, dy, dx]) * spacing
                    d, _ = tree.query(sub)
                    inside += d <= radius
        occupancy[shell] = inside / _SUBSAMPLES**3
    occupancy = occupancy.reshape(tuple(hi - lo))

    blurred = gaussian_filter(occupancy, sigma=optics / spacing)
    along = _values_along(blurred, line - lo * spacing, spacing)
    centre = float(np.median(along)) if along.size else 0.0
    if centre <= 0:
        return None
    signal = blurred * (float(peak) / centre)
    noise = rng.normal(0.0, 1.0, size=signal.shape) * np.sqrt(max(float(noise_gain), 0.0) * signal)
    volume[box] = np.maximum(np.asarray(volume[box], dtype=float) + signal + noise, 0.0)
    return box, occupancy >= 0.5


def _extended(line_um: np.ndarray, by_um: float) -> np.ndarray:
    """*line_um* carried on *by_um* past each end, along its end directions."""
    line = np.asarray(line_um, dtype=float)
    if len(line) < 2 or by_um <= 0:
        return line
    start_direction = line[0] - line[min(len(line) - 1, 3)]
    end_direction = line[-1] - line[max(0, len(line) - 4)]
    ends = []
    for point, direction in ((line[0], start_direction), (line[-1], end_direction)):
        norm = float(np.linalg.norm(direction))
        ends.append(point + (direction / norm) * by_um if norm > 0 else point)
    return np.vstack([ends[0], line, ends[1]])


def _positive(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def plant_vessels(
    raw_volume: np.ndarray,
    G: nx.MultiGraph,
    edges,
    vessel_mask: np.ndarray,
    voxel_size_zyx,
    *,
    image_psf_um: Optional[Sequence[float]],
    rng: np.random.Generator,
    sample_size: int = PLANTED_SAMPLE_SIZE,
    guide_attribute: str = "edt_diameter_um",
    copy_attributes: Sequence[str] = ("diameter_um",),
) -> Optional[PlantedVessels]:
    """Plant a vessel beside each of up to *sample_size* of *edges* of *G*
    (see the module docstring), in a copy of *raw_volume*; ``None`` when
    none fits. *image_psf_um* is the blur the image shows (see
    :func:`optics_psf`); *rng* makes it repeatable."""
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    optics = optics_psf(image_psf_um, voxel_size_zyx)
    edges = list(edges)
    gain = image_noise_gain(raw_volume, [G.edges[e]["voxels"] for e in edges], voxel_size_zyx)
    volume = np.array(raw_volume, dtype=np.float32, copy=True)
    occupied = np.array(vessel_mask, dtype=bool, copy=True)

    probe = nx.MultiGraph()
    probe.graph.update(G.graph)
    for edge in [e for e in edges for _copy in range(PLANTED_PER_VESSEL)]:
        if probe.number_of_edges() >= max(0, int(sample_size)):
            break
        source = G.edges[edge]
        guide = _positive(source.get(guide_attribute)) or 4.0
        width = guide * float(rng.uniform(1.0 - PLANTED_WIDTH_SPREAD, 1.0 + PLANTED_WIDTH_SPREAD))
        clearance = PLANTED_CLEARANCE_MULTIPLE * width + 2.0 * max(optics)
        placed = fwhm_decoys.decoy_centrelines(
            G, [edge], occupied, voxel_size_zyx, rng=rng, guide_attribute=guide_attribute,
            min_clearance_um=clearance,
        )
        if not placed:
            continue
        line = placed[0][1]
        # Drawn on past both ends where there is room: a lumen stopping where
        # its measured stretch does tapers in the blur there, and reads
        # narrow -- a real vessel carries on.
        drawn_line = _extended(line, width + 3.0 * max(optics))
        if not fwhm_decoys._clear_of_vessels(
            drawn_line[[0, -1]], occupied, spacing, clearance
        ):
            drawn_line = line
        real = _values_along(raw_volume, np.asarray(source["voxels"], dtype=float), spacing)
        here = _values_along(raw_volume, line, spacing)
        if real.size == 0 or here.size == 0:
            continue
        contrast = float(np.median(real)) - float(np.median(here))
        if contrast <= 0:
            continue
        drawn = draw_vessel(
            volume, drawn_line, width, voxel_size_zyx, optics, peak=contrast, noise_gain=gain, rng=rng
        )
        if drawn is None:
            continue
        box, lumen = drawn
        occupied[box] |= lumen
        attrs = {
            "voxels": [tuple(p) for p in line],
            "length": float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1))),
            "planted_diameter_um": width,
            guide_attribute: guide,
        }
        attrs.update({name: source[name] for name in copy_attributes if name in source})
        index = probe.number_of_edges()
        probe.add_edge(2 * index, 2 * index + 1, key=0, **attrs)
    if probe.number_of_edges() == 0:
        return None
    return PlantedVessels(volume=volume, probe=probe, optics_psf_um=optics, noise_gain=gain)


def planted_width_report(probe: nx.MultiGraph) -> dict[str, Any]:
    """How far FWHM's widths on a measured *probe* of planted vessels are
    from their true widths: ``relative_error`` is the mean over every
    planted vessel of ``|width - true| / true``, at most 1, an unmeasured one
    counting as 1 (all of its width wrong) -- a mean, so one vessel moving
    moves it a little rather than a median jumping between vessels;
    ``relative_error_measured`` the median over the measured ones alone."""
    errors, measured_errors = [], []
    for _u, _v, data in probe.edges(data=True):
        true = float(data["planted_diameter_um"])
        width = _positive(data.get("fwhm_diameter_um"))
        if width is None:
            errors.append(1.0)
            continue
        error = min(abs(width - true) / true, 1.0)
        errors.append(error)
        measured_errors.append(error)
    planted = len(errors)
    return dict(
        planted=planted,
        planted_measured=len(measured_errors),
        measured_fraction=float(len(measured_errors) / planted) if planted else 0.0,
        relative_error=float(np.mean(errors)) if errors else 1.0,
        relative_error_measured=float(np.median(measured_errors)) if measured_errors else None,
    )
