"""Is the ring fit right? Vessels of known lumen, planted in the run's own
endothelial image.

The counterpart of :mod:`.fwhm_planted` for an endothelial stain, which the
endothelial settings optimiser scores accuracy by: nothing in the image knows
a real vessel's true internal diameter, and coverage and the agreement of
readings along a vessel cannot tell a right width from a consistently wrong
one.

Beside each of a sample of real vessels, in tissue the segmentation says is
empty, a hollow vessel is drawn into a copy of the endothelial channel: the
real vessel's centreline moved sideways, a lumen of known diameter inside a
wall of the image's own thickness, blurred by the image's own PSF -- the two
as the image's rings calibrate them
(:func:`~haemolynx.haemodynamics.endothelial.calibrate_from_rings`) -- its
wall and lumen as bright as the real vessel's own ring fits them at that blur
and wall, with shot noise of the image's own scale. Its lumen is the real
vessel's, as its ring reads, give or take a quarter (so a reading guided by
the vessel's mask width does not start at the answer), and it is guided by
the real vessel's mask width scaled alike. Measured with a run's ring-fit
settings, the planted vessels say how far off its internal diameters are
(:func:`planted_lumen_report`).

The planted vessels carry the image's calibrated blur and wall, so they
cannot judge those two: a reading that holds them at other values reads the
planted vessels worse by construction, whatever the real ones show.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from . import endothelial, fwhm_decoys, fwhm_planted
from .fwhm_diameter import _arc_length_parameterize

__all__ = [
    "PLANTED_SAMPLE_SIZE",
    "PlantedRings",
    "REFERENCE_MIN_CONTRAST",
    "draw_ring",
    "plant_rings",
    "planted_lumen_report",
]

#: Vessels planted at most, and beside each sampled one: as for FWHM's
#: planted vessels (:mod:`.fwhm_planted`).
PLANTED_SAMPLE_SIZE = fwhm_planted.PLANTED_SAMPLE_SIZE
PLANTED_PER_VESSEL = fwhm_planted.PLANTED_PER_VESSEL
PLANTED_LUMEN_SPREAD = fwhm_planted.PLANTED_WIDTH_SPREAD
PLANTED_CLEARANCE_MULTIPLE = fwhm_planted.PLANTED_CLEARANCE_MULTIPLE

#: The weakest ring a real vessel's planted copies are drawn like: its wall
#: twice its fit's scatter above its lumen, half the ring fit's own default
#: gate, so the vessels a gate set too strict refuses still give rings to
#: draw.
REFERENCE_MIN_CONTRAST = 0.5 * endothelial.ENDOTHELIAL_MIN_RING_CONTRAST

#: Where along a real vessel its ring is read, as a share of its length.
_REFERENCE_POSITIONS = (0.5, 0.25, 0.75)

#: Sub-samples per voxel axis drawing the lumen's and the wall's edges, which
#: also puts the voxels' own box into the blur (see
#: :func:`~.fwhm_planted.optics_psf`); a wall is about a voxel thick.
_SUBSAMPLES = 4

#: A guide width for a vessel without one, as the ring fit's own fallback.
_FALLBACK_GUIDE_UM = 4.0

#: How far sideways a planted vessel is tried, nearest first, as multiples
#: of its clearance plus its real vessel's half width -- FWHM's fixed 8-20 um
#: leave a wide ring no room -- and in how many directions at each.
_PLACEMENT_DISTANCES = (1.0, 1.25, 1.5, 2.0)
_DIRECTIONS_PER_DISTANCE = 24


@dataclass(frozen=True)
class PlantedRings:
    """A copy of the endothelial channel with vessels drawn in, and a graph of them."""

    #: The channel, as float32, with every planted vessel added.
    volume: np.ndarray
    #: One edge per planted vessel, its centreline in ``voxels``, its true
    #: internal diameter in ``planted_diameter_um`` and its guide width as the
    #: guide attribute, so the ring fit reads it as it would a real vessel.
    probe: nx.MultiGraph
    #: The blur ``(z, y, x)`` sigma in um and the wall they were drawn with:
    #: the image's, as its rings calibrate them.
    psf_sigma_zyx: tuple[float, float, float]
    wall_um: float
    #: The optics' share of that blur (the voxels' box taken out).
    optics_psf_um: tuple[float, float, float]
    #: The shot-noise scale they were drawn with (variance per unit of signal).
    noise_gain: float
    #: How many of the sampled vessels showed a ring to draw them like.
    reference_rings: int


def _cover(distance: np.ndarray, inner: float, outer: float) -> np.ndarray:
    """1 where *inner* < distance <= *outer*."""
    return ((distance > inner) & (distance <= outer)).astype(float)


def draw_ring(
    volume: np.ndarray,
    centreline_um,
    lumen_diameter_um: float,
    wall_um: float,
    voxel_size_zyx,
    optics_psf_um,
    *,
    wall_level: float,
    lumen_level: float,
    noise_gain: float,
    rng: np.random.Generator,
) -> Optional[tuple[tuple[slice, ...], np.ndarray]]:
    """Add a hollow vessel along *centreline_um* to *volume*, in place: a
    lumen *lumen_diameter_um* across inside a wall *wall_um* thick, the wall
    *wall_level* and the lumen *lumen_level* above what was there before the
    blur (the ring fit's own model, ``RingFit.wall_level`` /
    ``lumen_level``), blurred by *optics_psf_um*, with Gaussian shot noise of
    variance *noise_gain* times the signal, and nothing below zero. Works in
    a box round the vessel only. Returns the box and the vessel's outer
    extent in it (where at least half of a voxel is inside), or ``None``
    when the vessel misses the volume."""
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    optics = np.asarray(optics_psf_um, dtype=float)
    radius = 0.5 * float(lumen_diameter_um)
    outer = radius + float(wall_um)
    line = fwhm_planted._dense(np.asarray(centreline_um, dtype=float), 0.25 * float(spacing.min()))
    margin = outer + 4.0 * float(optics.max()) + float(spacing.max())
    shape = np.asarray(volume.shape)
    lo = np.maximum(np.floor((line.min(axis=0) - margin) / spacing).astype(int), 0)
    hi = np.minimum(np.ceil((line.max(axis=0) + margin) / spacing).astype(int) + 1, shape)
    if np.any(hi <= lo):
        return None
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    centres = (np.stack(np.indices(tuple(hi - lo)), axis=-1).reshape(-1, 3) + lo) * spacing

    tree = cKDTree(line)
    half_diagonal = 0.5 * float(np.linalg.norm(spacing))
    distance, _ = tree.query(centres, distance_upper_bound=outer + half_diagonal)
    lumen = (distance <= radius - half_diagonal).astype(float)
    wall = _cover(distance, radius + half_diagonal, outer - half_diagonal)
    # A voxel within half its diagonal of either edge is partly in: its
    # share is counted on sub-voxel samples.
    shell = np.flatnonzero(
        np.isfinite(distance)
        & ((np.abs(distance - radius) < half_diagonal) | (np.abs(distance - outer) < half_diagonal))
    )
    if shell.size:
        offsets = (np.arange(_SUBSAMPLES) + 0.5) / _SUBSAMPLES - 0.5
        in_lumen = np.zeros(shell.size)
        in_wall = np.zeros(shell.size)
        for dz in offsets:
            for dy in offsets:
                for dx in offsets:
                    d, _ = tree.query(centres[shell] + np.array([dz, dy, dx]) * spacing)
                    in_lumen += d <= radius
                    in_wall += (d > radius) & (d <= outer)
        lumen[shell] = in_lumen / _SUBSAMPLES**3
        wall[shell] = in_wall / _SUBSAMPLES**3
    size = tuple(hi - lo)
    sigma = optics / spacing
    signal = float(wall_level) * gaussian_filter(wall.reshape(size), sigma=sigma) + float(
        lumen_level
    ) * gaussian_filter(lumen.reshape(size), sigma=sigma)
    noise = rng.normal(0.0, 1.0, size=signal.shape) * np.sqrt(
        max(float(noise_gain), 0.0) * np.maximum(signal, 0.0)
    )
    volume[box] = np.maximum(np.asarray(volume[box], dtype=float) + signal + noise, 0.0)
    return box, (lumen + wall).reshape(size) >= 0.5


def _placed_beside(
    data: dict, occupied: np.ndarray, spacing: np.ndarray, rng: np.random.Generator,
    clearance_um: float, guide_um: float, face_margin_um: float,
) -> Optional[np.ndarray]:
    """The vessel's centreline moved sideways -- perpendicular to its overall
    direction, nearest first (:data:`_PLACEMENT_DISTANCES`) -- to where it
    keeps *clearance_um* from every occupied voxel and *face_margin_um* from
    the volume's faces, or ``None``."""
    poly = np.asarray(data["voxels"], dtype=float)
    direction = poly[-1] - poly[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        return None
    direction /= norm
    far_face = (np.asarray(occupied.shape) - 1) * spacing - float(face_margin_um)
    reach = float(clearance_um) + 0.5 * float(guide_um)
    for multiple in _PLACEMENT_DISTANCES:
        for _attempt in range(_DIRECTIONS_PER_DISTANCE):
            side = rng.normal(size=3)
            side -= (side @ direction) * direction
            if np.linalg.norm(side) < 1e-9:
                continue
            moved = poly + multiple * reach * side / np.linalg.norm(side)
            if not np.all((moved >= face_margin_um) & (moved <= far_face)):
                continue
            if fwhm_decoys._clear_of_vessels(moved, occupied, spacing, clearance_um):
                return moved
    return None


def _guide(data: dict, guide_attribute: str) -> float:
    return fwhm_planted._positive(data.get(guide_attribute)) or _FALLBACK_GUIDE_UM


def _reference_ring(
    raw_volume: np.ndarray,
    data: dict,
    voxel_size_zyx,
    psf_sigma_zyx,
    wall_um: float,
    guide_um: float,
) -> Optional[tuple[float, float, float]]:
    """``(internal diameter, wall level, lumen level)`` the vessel's own ring
    reads at the image's blur and wall -- the median over the sections along
    it that show a ring -- or ``None`` when none does."""
    poly = np.asarray(data["voxels"], dtype=float)
    if len(poly) < 2:
        return None
    s, total = _arc_length_parameterize(poly)
    if total <= 0:
        return None
    readings = []
    for share in _REFERENCE_POSITIONS:
        fit, _why = endothelial.fit_ring_section(
            raw_volume, poly, s, total, share * total, voxel_size_zyx, guide_um=guide_um,
            psf_sigma_zyx=psf_sigma_zyx, wall_um=wall_um,
        )
        if fit is not None and endothelial._gate(
            fit, min_contrast=REFERENCE_MIN_CONTRAST, min_diameter_um=0.0
        ) == "ok":
            readings.append((fit.diameter_um, fit.wall_level, fit.lumen_level))
    if not readings:
        return None
    return tuple(float(v) for v in np.median(np.asarray(readings), axis=0))


def plant_rings(
    raw_volume: np.ndarray,
    G: nx.MultiGraph,
    edges,
    vessel_mask: np.ndarray,
    voxel_size_zyx,
    *,
    psf_sigma_zyx: Sequence[float],
    wall_um: float,
    rng: np.random.Generator,
    sample_size: int = PLANTED_SAMPLE_SIZE,
    guide_attribute: str = "edt_diameter_um",
) -> Optional[PlantedRings]:
    """Plant a hollow vessel beside each of up to *sample_size* of *edges* of
    *G* (see the module docstring), in a copy of *raw_volume*, the
    endothelial channel; ``None`` when no sampled vessel shows a ring to draw
    them like, or none fits. *psf_sigma_zyx* and *wall_um* are the blur and
    wall the image's rings show (what a fit sees: the voxels' box included);
    *rng* makes it repeatable."""
    spacing = np.asarray(voxel_size_zyx, dtype=float)
    psf = tuple(float(v) for v in psf_sigma_zyx)
    wall_um = float(wall_um)
    edges = list(edges)
    references = {
        edge: _reference_ring(
            raw_volume, G.edges[edge], voxel_size_zyx, psf, wall_um,
            _guide(G.edges[edge], guide_attribute),
        )
        for edge in edges
    }
    found = [
        (reference, _guide(G.edges[edge], guide_attribute))
        for edge, reference in references.items()
        if reference is not None
    ]
    if not found:
        return None
    # A vessel whose own ring does not show is drawn like the ones that do:
    # its lumen in the same proportion to its guide, as bright.
    typical_ratio = float(np.median([reference[0] / guide for reference, guide in found]))
    typical_levels = tuple(float(v) for v in np.median([reference[1:] for reference, _g in found], axis=0))

    optics = fwhm_planted.optics_psf(psf, voxel_size_zyx)
    gain = fwhm_planted.image_noise_gain(
        raw_volume, [G.edges[e]["voxels"] for e in edges], voxel_size_zyx
    )
    volume = np.array(raw_volume, dtype=np.float32, copy=True)
    occupied = np.array(vessel_mask, dtype=bool, copy=True)

    probe = nx.MultiGraph()
    probe.graph.update(G.graph)
    for edge in [e for e in edges for _copy in range(PLANTED_PER_VESSEL)]:
        if probe.number_of_edges() >= max(0, int(sample_size)):
            break
        guide = _guide(G.edges[edge], guide_attribute)
        reference = references[edge]
        lumen, (wall_level, lumen_level) = (
            (typical_ratio * guide, typical_levels) if reference is None
            else (reference[0], reference[1:])
        )
        factor = float(rng.uniform(1.0 - PLANTED_LUMEN_SPREAD, 1.0 + PLANTED_LUMEN_SPREAD))
        lumen *= factor
        outer_diameter = lumen + 2.0 * wall_um
        clearance = PLANTED_CLEARANCE_MULTIPLE * outer_diameter + 2.0 * max(optics)
        # The ring fit reads a window three blurs past the wall: a vessel
        # nearer a face than that would be read off the image.
        face_margin = 0.5 * outer_diameter + 3.0 * max(psf) + 1.0
        line = _placed_beside(G.edges[edge], occupied, spacing, rng, clearance, guide, face_margin)
        if line is None:
            continue
        # Drawn on past both ends where there is room, as FWHM's planted
        # vessels are: a lumen stopping where its read stretch does would
        # taper in the blur there.
        drawn_line = fwhm_planted._extended(line, outer_diameter + 3.0 * max(optics))
        if not fwhm_decoys._clear_of_vessels(drawn_line[[0, -1]], occupied, spacing, clearance):
            drawn_line = line
        drawn = draw_ring(
            volume, drawn_line, lumen, wall_um, voxel_size_zyx, optics,
            wall_level=wall_level, lumen_level=lumen_level, noise_gain=gain, rng=rng,
        )
        if drawn is None:
            continue
        box, extent = drawn
        occupied[box] |= extent
        index = probe.number_of_edges()
        probe.add_edge(
            2 * index, 2 * index + 1, key=0,
            voxels=[tuple(p) for p in line],
            length=float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1))),
            planted_diameter_um=lumen,
            **{guide_attribute: guide * factor},
        )
    if probe.number_of_edges() == 0:
        return None
    return PlantedRings(
        volume=volume, probe=probe, psf_sigma_zyx=psf, wall_um=wall_um, optics_psf_um=optics,
        noise_gain=gain, reference_rings=len(found),
    )


def planted_lumen_report(probe: nx.MultiGraph) -> dict[str, Any]:
    """How far the ring fit's internal diameters on a measured *probe* of
    planted vessels are from their true ones, as
    :func:`~.fwhm_planted.planted_width_report` says it for FWHM:
    ``relative_error`` the mean over every planted vessel of ``|width -
    true| / true``, at most 1, an unmeasured one counting as 1;
    ``relative_error_measured`` the median over the measured ones alone."""
    errors, measured_errors = [], []
    for _u, _v, data in probe.edges(data=True):
        true = float(data["planted_diameter_um"])
        width = (
            fwhm_planted._positive(data.get("endothelial_diameter_um"))
            if data.get("endothelial_status") == "measured" else None
        )
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
