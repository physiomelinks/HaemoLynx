"""The tissue's own volume and surface, measured from the raw image.

Vessel density is vessel per unit of *tissue*, but much of a stack is often not
tissue at all: empty space round a slice or an explant, at a lower background
than the tissue's own (autofluorescence, a little dye in the parenchyma).
Dividing by the whole image -- or by the box the network spans -- counts that
space as tissue and understates the density by however much of the image it
fills.

This finds the tissue as the region where the raw image's *background* is
high, and measures it as a closed surface:

1. **Block averages** of the raw image onto a working grid of about
   ``working_voxel_size_um`` (whole blocks of voxels per axis, never coarser
   than asked; a partial block at a far face averages only the voxels it has),
   leaving out the vessel voxels -- the run's own segmentation. Vessels are
   the brightest thing in the image and would otherwise decide a threshold
   that is meant to separate tissue from empty space.
2. **A split** between empty space and tissue in those averages, by an
   automatic threshold (Otsu, Li or triangle) or a manual intensity. An
   automatic split does not place the surface: the surface sits halfway
   between the two classes' median levels. A tissue edge is a blurred step,
   and a blurred step crosses its halfway level where the step is, whatever
   the blur -- an automatic split, pulled towards one class or the other, does
   not. A manual intensity is the level itself.
3. **Smoothing** by normalised convolution: the Gaussian-smoothed sum of
   intensity over the Gaussian-smoothed voxel count, with every vessel voxel
   read as the tissue's own level (a vessel is tissue), so each point holds
   the local tissue background and no voxel outside the image counts.
4. **Clean-up** on the working grid: pieces smaller than
   ``min_component_fraction`` of the largest are dropped, and enclosed
   cavities are filled.
5. **The surface**: marching cubes at that level through the smoothed field --
   clamped to the cleaned mask, so the clean-up is honoured -- padded so it
   closes where the image's own faces cut the tissue, with those caps placed
   exactly on the faces. Its enclosed volume is the tissue volume: sub-voxel,
   where a voxel count is off by up to half a working voxel at every face.

The image faces are part of the closed surface but not of the tissue's own:
``surface_area_um2`` leaves them out and ``cut_face_area_um2`` reports them.
Vertices are physical ``(z, y, x)`` microns, the frame node ``pos`` is in, so
the surface overlays the network with ``scale=(1, 1, 1)``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: The automatic splits, and ``"manual"`` for an intensity the user gives.
TISSUE_THRESHOLD_METHODS: Tuple[str, ...] = ("otsu", "li", "triangle", "manual")

#: Below this, the two intensity classes overlap too much for the split to be
#: tissue against empty space (Ashman's D: the gap between the class levels
#: over their pooled spread). An image that is tissue from edge to edge has
#: one class, and an automatic split then cuts the tissue itself in two --
#: which scores about 2.3 for one Gaussian class (each half's spread is
#: truncated by the split) and 2.7 for a flat one, against tens for tissue
#: beside empty space.
MIN_CLASS_SEPARATION = 3.0

#: Above this share of vessel, what was found is not tissue: vessels are a few
#: percent of brain and well under a third of any tissue. A "tissue" this
#: vascular is the vessels' own blur, found in a channel that shows no tissue
#: background at all (a dextran channel is black outside the vessels).
MAX_PLAUSIBLE_VESSEL_FRACTION = 0.3

#: A vertex this close (in working voxels) to a cap plane lies on it. Marching
#: cubes places vertices in float32; no vertex off a cap comes anywhere near
#: one, so the tolerance can be generous.
_CAP_TOLERANCE = 1e-3


@dataclass(frozen=True, eq=False)
class TissueVolume:
    """The tissue a raw image holds, measured as a closed surface."""

    #: Enclosed by the surface: the tissue volume, in cubic microns.
    volume_um3: float
    #: The cleaned working-grid mask, each block counted at its own size: a
    #: cross-check on ``volume_um3``.
    voxel_volume_um3: float
    #: The tissue's own surface, leaving out where the image faces cut it (um²).
    surface_area_um2: float
    #: Where the image faces cut the tissue (um²).
    cut_face_area_um2: float
    #: The whole image (um³).
    image_volume_um3: float
    #: Segmented vessel voxels inside the tissue, or in the working blocks its
    #: surface passes through (um³); 0 without a mask.
    vessel_volume_um3: float
    #: The intensity the surface passes through, in raw image units.
    level: float
    #: The automatic split, or the manual intensity, the level came from.
    split: float
    #: Median smoothed intensity of the empty-space and the tissue class.
    background_level: float
    tissue_level: float
    #: Ashman's D between the two classes; see :data:`MIN_CLASS_SEPARATION`.
    class_separation: float
    threshold_method: str
    #: The working grid's voxel size, ``(z, y, x)`` microns.
    working_voxel_size_zyx: Tuple[float, float, float]
    #: Separate pieces of tissue kept.
    component_count: int
    #: Surface vertices in physical ``(z, y, x)`` microns, and the triangles
    #: indexing them.
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def tissue_fraction(self) -> float:
        """The share of the image that is tissue."""
        if self.image_volume_um3 <= 0:
            return 0.0
        return self.volume_um3 / self.image_volume_um3

    def rows(self) -> Dict[str, Any]:
        """The measurement as statistics rows, ``"Name (unit)": value``."""
        z, y, x = self.working_voxel_size_zyx
        return {
            "Tissue Volume (micron³)": self.volume_um3,
            "Tissue Volume, Voxel Count (micron³)": self.voxel_volume_um3,
            "Tissue Fraction of Image": self.tissue_fraction,
            "Tissue Surface Area (micron²)": self.surface_area_um2,
            "Tissue Surface Cut by Image Faces (micron²)": self.cut_face_area_um2,
            "Tissue Pieces": self.component_count,
            "Vessel Volume in Tissue (micron³)": self.vessel_volume_um3,
            "Tissue Threshold Method": self.threshold_method,
            "Tissue Threshold Split": self.split,
            "Tissue Surface Intensity": self.level,
            "Empty-Space Intensity": self.background_level,
            "Tissue Intensity": self.tissue_level,
            "Tissue/Empty-Space Separation": self.class_separation,
            "Tissue Working Voxel Size (z, y, x microns)": f"{z:g} x {y:g} x {x:g}",
        }


def measure_tissue_volume(
    raw: np.ndarray,
    voxel_size_zyx: Sequence[float],
    *,
    vessel_mask: Optional[np.ndarray] = None,
    working_voxel_size_um: float = 4.0,
    smoothing_sigma_um: float = 5.0,
    threshold_method: str = "otsu",
    manual_threshold: Optional[float] = None,
    min_component_fraction: float = 0.01,
    fill_holes: bool = True,
) -> TissueVolume:
    """Measure the tissue in *raw* as a closed surface (see the module notes).

    *raw* is the intensity volume in canonical ``(z, y, x)`` order -- a
    memory-mapped one is read a plane at a time. *vessel_mask*, on the same
    grid, is the segmentation: left out of the threshold, and read as tissue at
    the tissue's own level. *voxel_size_zyx* is per-array-axis spacing in microns.
    """
    from scipy import ndimage

    if raw.ndim != 3:
        raise ValueError(f"Expected a 3D (z, y, x) raw volume, got shape {tuple(raw.shape)}.")
    if vessel_mask is not None and tuple(vessel_mask.shape) != tuple(raw.shape):
        raise ValueError(
            f"The vessel mask is {tuple(vessel_mask.shape)} but the raw image is "
            f"{tuple(raw.shape)}; the tissue volume needs the raw image on the "
            "segmentation's own grid."
        )
    if threshold_method not in TISSUE_THRESHOLD_METHODS:
        raise ValueError(
            f"threshold_method must be one of {TISSUE_THRESHOLD_METHODS}, "
            f"got {threshold_method!r}."
        )
    if threshold_method == "manual" and manual_threshold is None:
        raise ValueError("threshold_method='manual' needs a manual_threshold.")
    spacing = tuple(float(v) for v in voxel_size_zyx)
    if len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError(f"voxel_size_zyx must be three positive spacings, got {voxel_size_zyx}.")

    shape = tuple(int(n) for n in raw.shape)
    factors = _block_factors(shape, spacing, working_voxel_size_um)
    working_spacing = tuple(f * s for f, s in zip(factors, spacing))
    sums, counts, vessels = _block_sums(raw, vessel_mask, factors)
    block_voxels = np.einsum(
        "i,j,k->ijk", *(_block_extents(n, f) for n, f in zip(shape, factors))
    ).astype(np.float64)

    # The classes, and the levels the surface sits halfway between, come from
    # the unsmoothed block averages of the non-vessel voxels: smoothing spreads
    # every edge over a few sigma, and on a small or thin piece of tissue those
    # edge blocks are enough of the whole to drag both class medians together.
    has_data = counts > 0
    split, level, background_level, tissue_level, separation = _surface_level(
        sums[has_data] / counts[has_data], threshold_method, manual_threshold
    )
    # A vessel voxel is tissue, so it reads as the tissue's own level -- not
    # its brightness, and not nothing: leaving it out of a block on the
    # tissue's edge would leave that block's empty-space half to speak for
    # it, and pull the edge inward by however much of the tissue is vessel.
    sums = sums + tissue_level * vessels
    counts = counts + vessels
    if smoothing_sigma_um > 0:
        sigma = [smoothing_sigma_um / s for s in working_spacing]
        sums = ndimage.gaussian_filter(sums, sigma, mode="constant", cval=0.0)
        counts = ndimage.gaussian_filter(counts, sigma, mode="constant", cval=0.0)
    # Every block holds at least one voxel, so no smoothed count is zero.
    field = sums / counts
    if threshold_method != "manual" and separation < MIN_CLASS_SEPARATION:
        logger.warning(
            "Tissue volume: the raw image's empty space and tissue barely differ "
            "(separation %.2f, below %.1f), so the %s split may be cutting the "
            "tissue itself rather than finding its edge -- an image that is tissue "
            "from edge to edge has no empty space to find. Check the tissue surface, "
            "or set tissue_threshold_method to manual.",
            separation, MIN_CLASS_SEPARATION, threshold_method,
        )

    mask = field >= level
    mask, component_count = _keep_large_pieces(mask, min_component_fraction)
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)

    # The surface runs through the blocks just outside the mask too, so the
    # vessel inside it is counted over those as well.
    touching = ndimage.binary_dilation(mask, structure=np.ones((3, 3, 3), dtype=bool))
    vessel_volume = float(vessels[touching].sum()) * float(np.prod(spacing))
    voxel_volume = float(block_voxels[mask].sum()) * float(np.prod(spacing))
    image_volume = float(np.prod([n * s for n, s in zip(shape, spacing)]))

    vertices, faces, on_cap = _surface(field, mask, level, shape, factors, spacing)
    areas = _triangle_areas(vertices, faces)
    tissue = TissueVolume(
        volume_um3=_enclosed_volume(vertices, faces),
        voxel_volume_um3=voxel_volume,
        surface_area_um2=float(areas[~on_cap].sum()),
        cut_face_area_um2=float(areas[on_cap].sum()),
        image_volume_um3=image_volume,
        vessel_volume_um3=vessel_volume,
        level=level,
        split=split,
        background_level=background_level,
        tissue_level=tissue_level,
        class_separation=separation,
        threshold_method=threshold_method,
        working_voxel_size_zyx=working_spacing,
        component_count=component_count,
        vertices=vertices,
        faces=faces,
    )
    if tissue.vessel_volume_um3 > MAX_PLAUSIBLE_VESSEL_FRACTION * tissue.volume_um3:
        logger.warning(
            "Tissue volume: %.0f%% of the 'tissue' found is vessel, more than any "
            "tissue holds -- the raw image seems to show no tissue background (a "
            "plasma or dextran channel is dark outside the vessels), so what was "
            "found is the vessels' own blur. Use a channel where the tissue itself "
            "is brighter than the empty space (tissue_raw_channel).",
            100.0 * tissue.vessel_volume_um3 / max(tissue.volume_um3, 1e-300),
        )
    logger.info(
        "Tissue volume: %.6g um^3 (%.1f%% of the image), surface %.6g um^2 "
        "(plus %.6g um^2 cut by the image faces), %d piece(s); surface at "
        "intensity %.6g between empty space %.6g and tissue %.6g.",
        tissue.volume_um3, 100.0 * tissue.tissue_fraction, tissue.surface_area_um2,
        tissue.cut_face_area_um2, component_count, level, background_level, tissue_level,
    )
    return tissue


def _block_factors(
    shape: Tuple[int, ...], spacing: Tuple[float, ...], working_voxel_size_um: float
) -> Tuple[int, int, int]:
    """Voxels per working block along each axis: as many as fit in
    *working_voxel_size_um*, at least one, at most the axis."""
    factors = []
    for n, s in zip(shape, spacing):
        f = int(np.floor(working_voxel_size_um / s + 1e-9)) if working_voxel_size_um > 0 else 1
        factors.append(int(min(max(f, 1), n)))
    return tuple(factors)


def _block_extents(n: int, f: int) -> np.ndarray:
    """How many voxels each block along an axis of *n* holds, in blocks of *f*."""
    starts = np.arange(0, n, f)
    return np.minimum(starts + f, n) - starts


def _block_sums(raw, vessel_mask, factors):
    """Per working block: summed non-vessel intensity, non-vessel voxel count
    and vessel voxel count. One z-plane at a time, so a memory-mapped volume
    is never read whole."""
    nz, ny, nx = raw.shape
    fz, fy, fx = factors
    y_starts, x_starts = np.arange(0, ny, fy), np.arange(0, nx, fx)
    grid = (len(range(0, nz, fz)), len(y_starts), len(x_starts))
    sums = np.zeros(grid, dtype=np.float64)
    counts = np.zeros(grid, dtype=np.float64)
    vessels = np.zeros(grid, dtype=np.float64)

    def blocks(plane: np.ndarray) -> np.ndarray:
        return np.add.reduceat(np.add.reduceat(plane, y_starts, axis=0), x_starts, axis=1)

    for z in range(nz):
        plane = np.asarray(raw[z], dtype=np.float64)
        row = z // fz
        if vessel_mask is None:
            sums[row] += blocks(plane)
            counts[row] += blocks(np.ones(plane.shape))
            continue
        in_vessel = np.asarray(vessel_mask[z]) != 0
        sums[row] += blocks(np.where(in_vessel, 0.0, plane))
        counts[row] += blocks((~in_vessel).astype(np.float64))
        vessels[row] += blocks(in_vessel.astype(np.float64))
    return sums, counts, vessels


def _automatic_split(values: np.ndarray, method: str) -> float:
    from skimage import filters

    function = {
        "otsu": filters.threshold_otsu,
        "li": filters.threshold_li,
        "triangle": filters.threshold_triangle,
    }[method]
    return float(function(values))


def _robust_sigma(values: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def _surface_level(
    values: np.ndarray, method: str, manual_threshold: Optional[float]
) -> Tuple[float, float, float, float, float]:
    """``(split, level, background, tissue, separation)``: where the classes
    divide, the intensity the surface passes through, each class's median and
    how far apart they are (Ashman's D, on robust spreads)."""
    if values.size == 0:
        # Vessel from face to face: nothing but tissue.
        return 0.0, 0.0, 0.0, 0.0, float("inf")
    low_value, high_value = float(values.min()), float(values.max())
    if method == "manual":
        split = float(manual_threshold)
    elif high_value <= low_value:
        split = low_value
    else:
        split = _automatic_split(values, method)
    background = values[values < split]
    tissue = values[values >= split]
    if background.size == 0 or tissue.size == 0:
        # One class only: the level is the split, and nothing separates.
        only = tissue if tissue.size else background
        level_value = float(np.median(only))
        return split, split, level_value, level_value, 0.0
    background_level = float(np.median(background))
    tissue_level = float(np.median(tissue))
    spread = np.sqrt(0.5 * (_robust_sigma(background) ** 2 + _robust_sigma(tissue) ** 2))
    gap = tissue_level - background_level
    separation = float("inf") if spread == 0 else float(gap / spread)
    level = split if method == "manual" else 0.5 * (background_level + tissue_level)
    return split, level, background_level, tissue_level, separation


def _keep_large_pieces(mask: np.ndarray, min_component_fraction: float) -> Tuple[np.ndarray, int]:
    """*mask* without the pieces smaller than *min_component_fraction* of the
    largest, and how many pieces are left."""
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return mask, 0
    sizes = np.bincount(labels.ravel())[1:]
    keep = sizes >= float(min_component_fraction) * sizes.max()
    lookup = np.concatenate([[False], keep])
    return lookup[labels], int(keep.sum())


def _padded_axis_positions(n: int, f: int, spacing: float) -> np.ndarray:
    """Physical position of each padded working-grid index along one axis.

    Indices ``2..m+1`` are the blocks, at their own centres (a partial last
    block at its own). The two padding layers either side both sit on the
    image's face: the inner one carries the face blocks out to the face, and
    the surface closes between it and the outer one -- on the face itself."""
    starts = np.arange(0, n, f)
    extents = np.minimum(starts + f, n) - starts
    centres = (starts + (extents - 1) / 2.0) * spacing
    first_face, last_face = -0.5 * spacing, (n - 0.5) * spacing
    return np.concatenate([[first_face, first_face], centres, [last_face, last_face]])


def _surface(field, mask, level, shape, factors, spacing):
    """``(vertices, faces, on_cap)``: the closed surface of *mask* through
    *field* at *level*, vertices in physical ``(z, y, x)`` microns, and which
    triangles are caps on the image's faces."""
    from skimage.measure import marching_cubes

    empty = (np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64),
             np.empty(0, dtype=bool))
    if not mask.any():
        return empty
    delta = 1e-6 * max(abs(level), float(np.abs(field).max()), 1.0)
    # Through the field where the mask agrees with it, which is what places the
    # surface between working voxels; pinned just either side of the level
    # where the clean-up overruled it.
    iso = np.where(mask, np.maximum(field, level + delta), np.minimum(field, level - delta))
    # Two layers of padding, both placed on the image face (see
    # _padded_axis_positions). The inner one repeats the face blocks, so the
    # tissue's own surface runs straight out to the face instead of being
    # bevelled across the last half block; the outer one is mirrored about the
    # level, so the surface closes between the two -- on the face.
    inner = np.pad(iso, 1, mode="edge")
    padded = np.minimum(2.0 * level - np.pad(inner, 1, mode="edge"), level - delta)
    padded[1:-1, 1:-1, 1:-1] = inner
    grid_vertices, faces, _normals, _values = marching_cubes(
        padded, level=level, allow_degenerate=False
    )
    grid_vertices = grid_vertices.astype(np.float64)
    vertices = np.empty_like(grid_vertices)
    on_cap = np.zeros(len(faces), dtype=bool)
    for axis, (n, f, s) in enumerate(zip(shape, factors, spacing)):
        positions = _padded_axis_positions(n, f, s)
        vertices[:, axis] = np.interp(
            grid_vertices[:, axis], np.arange(positions.size, dtype=np.float64), positions
        )
        corner = grid_vertices[faces, axis]
        for plane in (0.5, positions.size - 1.5):
            on_cap |= np.all(np.abs(corner - plane) < _CAP_TOLERANCE, axis=1)
    # The walls between the two padding layers have no width once both sit on
    # the face: drop them, and the vertices only they used.
    kept = _triangle_areas(vertices, faces) > 0.0
    faces, on_cap = faces[kept], on_cap[kept]
    used, faces = np.unique(faces, return_inverse=True)
    return vertices[used], faces.reshape(-1, 3).astype(np.int64), on_cap


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.empty(0, dtype=np.float64)
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _enclosed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """The volume a closed, consistently wound triangle mesh encloses
    (divergence theorem), about the mesh's own centre to keep the triple
    products small."""
    if len(faces) == 0:
        return 0.0
    centred = vertices - vertices.mean(axis=0)
    a, b, c = (centred[faces[:, i]] for i in range(3))
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)
