"""Detect braided (medial-sheet) thick-vessel skeleton segments.

``skeletonize_thickness_gated`` builds a centreline *tree* for the fat part
of a vessel specifically to avoid the failure mode of naive Lee thinning on a
thick, blob-like region: instead of one centreline, Lee thinning of a fat
object often returns a medial *sheet* -- several near-parallel or crossing
polylines braided together, because a thick cylinder's skeleton is not
uniquely defined the way a thin one's is. ``preprocessing.thick_vessels``
already carries the metric for this (:func:`haemolynx.preprocessing.braid_factor`,
mean skeleton voxels per occupied slice along the vessel's own axis -- ~1 for
a single centreline, several for a sheet) and the threshold it is judged
against (:data:`haemolynx.preprocessing.BRAID_FACTOR_LIMIT`), but only as a
test-fixture characterisation tool, never applied to what a real run actually
produces.

This module applies that same metric to a finished run's own output, one fat
catchment component at a time. A long, round trunk -- the case this feature
is already known to get right -- resolves to a clean tree (braid factor
close to 1) under this check too, on both a real elongated tube and a short
one; a flattened, ribbon-like cross-section does not, on either a real
fused-capillary fixture or an isolated one built to rule out branching as
the cause -- an intrinsically ambiguous medial line, not a bug in the
tree-building, but exactly the shape this guard is for catching. Flagging by
component rather than as one score for the whole image is what makes "a good
trunk next to a bad side branch" visible instead of averaging it away.

Nothing here changes how a skeleton is built. It is a pure, read-only
diagnostic -- like ``graph.cartwheel_guard`` -- run against a finished
thickness-gated skeleton whenever you want to check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import find_objects, generate_binary_structure, label

from .thick_vessels import BRAID_FACTOR_LIMIT, braid_factor

__all__ = [
    "BraidedThickVesselComponent",
    "component_long_axis",
    "detect_braided_thick_vessel_components",
    "format_braided_thick_vessel_report",
]

#: A component with fewer occupied slices along its own long axis than this
#: cannot give braid_factor a stable mean -- one or two slices decide it
#: outright, and are usually a stub end rather than a genuine sheet.
DEFAULT_MIN_OCCUPIED_SLICES = 5


@dataclass(frozen=True)
class BraidedThickVesselComponent:
    """One fat-catchment component whose skeleton braids instead of a tree."""

    label: int
    long_axis: int
    braid_factor: float
    voxel_count: int
    occupied_slices: int
    #: (min, max) voxel index per axis, inclusive-exclusive like a slice.
    bounding_box: tuple[tuple[int, int], ...]
    #: Component centroid in physical microns, (z, y, x).
    centroid_um: tuple[float, float, float]


def component_long_axis(mask: np.ndarray) -> int:
    """Which array axis (0, 1 or 2) *mask*'s foreground is most elongated along.

    The principal axis of the foreground's own coordinates (largest-eigenvalue
    eigenvector of their covariance), matched to whichever canonical axis
    agrees with it most. A general-purpose orientation estimate -- not used by
    :func:`detect_braided_thick_vessel_components` itself, which tries every
    axis rather than trusting one guess (see its own docstring for why: a
    vessel much wider than the trunk length in view, an elliptical
    cross-section case a real fixture exposed, gets this wrong).
    """
    coords = np.argwhere(mask)
    if len(coords) < 2:
        return 0
    centred = coords - coords.mean(axis=0)
    covariance = np.cov(centred, rowvar=False)
    if covariance.shape != (3, 3):
        return 0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    return int(np.argmax(np.abs(principal)))


def detect_braided_thick_vessel_components(
    thick_vessel_mask: np.ndarray,
    skeleton: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    braid_factor_limit: float = BRAID_FACTOR_LIMIT,
    min_occupied_slices: int = DEFAULT_MIN_OCCUPIED_SLICES,
) -> list[BraidedThickVesselComponent]:
    """Every connected fat-catchment component whose own skeleton braids.

    *thick_vessel_mask* is the fat/thick catchment
    ``skeletonize_thickness_gated(..., return_thick_mask=True)`` already
    hands back; *skeleton* is that same call's tree. :func:`braid_factor`
    only measures braiding correctly when its axis runs along the vessel, not
    across it, and there is no reliable way to guess that axis up front -- a
    vessel with a wide, flattened cross-section can have more spatial extent
    across its width than along the length actually in view, which is
    exactly backwards from what a principal-axis estimate assumes (a real
    fused-trunk-plus-capillaries test fixture hits exactly this). So this
    tries all three canonical axes and takes whichever gives the *lowest*
    (least braided) reading among those with enough occupied slices to judge
    fairly (see *min_occupied_slices*) -- the most charitable reading is the
    right one to hold a component to, since a genuine single centreline reads
    close to 1 along its own axis regardless of which of the three canonical
    axes happens to be closest to it, while a real medial sheet reads high on
    every axis. A component with no axis clearing *min_occupied_slices* is
    skipped rather than guessed at either way. Returned worst-first (highest
    braid factor first).
    """
    if braid_factor_limit <= 0.0:
        raise ValueError("braid_factor_limit must be > 0")
    if min_occupied_slices < 1:
        raise ValueError("min_occupied_slices must be >= 1")

    mask = np.asarray(thick_vessel_mask, dtype=bool)
    skel = np.asarray(skeleton, dtype=bool)
    if mask.shape != skel.shape:
        raise ValueError(
            f"thick_vessel_mask and skeleton must share a shape, got "
            f"{mask.shape} and {skel.shape}."
        )
    if not mask.any():
        return []

    structure = generate_binary_structure(mask.ndim, mask.ndim)
    labeled, n_labels = label(mask, structure=structure)
    scale = np.asarray(voxel_size_zyx, dtype=float)

    flagged: list[BraidedThickVesselComponent] = []
    for component_id, bbox in enumerate(find_objects(labeled), start=1):
        if bbox is None:
            continue
        component_mask = labeled[bbox] == component_id
        component_skeleton = skel[bbox] & component_mask

        best_axis: int | None = None
        best_factor = float("inf")
        best_occupied = 0
        for axis in range(component_skeleton.ndim):
            counts = component_skeleton.sum(
                axis=tuple(i for i in range(component_skeleton.ndim) if i != axis)
            )
            occupied_count = int((counts > 0).sum())
            if occupied_count < min_occupied_slices:
                continue
            # counts/occupied_count above are only the gate; braid_factor is
            # the one place the actual number is computed, so both agree with
            # every other reading of it in this codebase (e.g. lee_braid_factor).
            factor = braid_factor(component_skeleton, axis=axis)
            if factor < best_factor:
                best_axis, best_factor, best_occupied = axis, factor, occupied_count

        if best_axis is None or best_factor <= braid_factor_limit:
            continue

        coords = np.argwhere(component_mask)
        centroid_voxels = coords.mean(axis=0) + np.array(
            [s.start for s in bbox], dtype=float
        )
        flagged.append(
            BraidedThickVesselComponent(
                label=component_id,
                long_axis=best_axis,
                braid_factor=best_factor,
                voxel_count=int(component_mask.sum()),
                occupied_slices=best_occupied,
                bounding_box=tuple(
                    (int(s.start), int(s.stop)) for s in bbox
                ),
                centroid_um=tuple(float(v) for v in centroid_voxels * scale),
            )
        )

    flagged.sort(key=lambda component: component.braid_factor, reverse=True)
    return flagged


def format_braided_thick_vessel_report(
    components: Sequence[BraidedThickVesselComponent],
) -> str:
    """A compact multiline report, in the style of format_cartwheel_hub_report."""
    if not components:
        return "Thick-vessel braid guard: no braided components flagged."
    lines = [
        f"Thick-vessel braid guard: {len(components)} component(s) flagged."
    ]
    for component in components:
        lines.append(
            f"  component={component.label}: braid_factor={component.braid_factor:.2f}, "
            f"voxels={component.voxel_count}, long_axis={component.long_axis}, "
            f"centroid_um={tuple(round(v, 1) for v in component.centroid_um)}, "
            f"bounding_box={component.bounding_box}"
        )
    return "\n".join(lines)
