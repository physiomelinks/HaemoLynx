"""How well a skeleton represents the mask it was extracted from.

A companion diagnostic to graph.diagnostics' skeleton-vs-graph consistency
check, which looks one stage later, at the graph against the skeleton it was
built from -- this one looks at the skeleton against the binary mask it came
from, before any graph topology exists to compare it against. Both exist to
surface skeletonisation/topology-repair quality loss in the GUI log instead
of it only showing up later as an unexplained missing vessel.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label
from scipy.ndimage import sum as ndi_sum

from .thick_vessels import inscribed_radius_map


def _explained_by_local_radius(
    source_bool: np.ndarray,
    mask_bool: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    local_radius_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Which *mask_bool* voxels count as "explained" by *source_bool*.

    A mask voxel is explained when the nearest True voxel of *source_bool*
    is no farther from it than that point's own local inscribed radius
    (:func:`thick_vessels.inscribed_radius_map`), plus one voxel diagonal's
    worth of grid-discretisation slack -- see
    :func:`diagnose_skeleton_mask_consistency`'s own docstring for the
    empirical justification of that margin.

    Shared by :func:`diagnose_skeleton_mask_consistency` (*source_bool* is
    the skeleton) and ``graph.diagnostics.diagnose_graph_mask_consistency``
    (*source_bool* is the graph's own rasterised edges) -- both compare
    against the same kind of segmented-image mask with the identical
    local-radius-plus-margin geometry, so a fix to that geometry (like the
    margin itself) only has to be made once.

    *local_radius_map*, when given, is used as-is instead of being
    recomputed from *mask_bool* -- a caller that already has
    :func:`thick_vessels.inscribed_radius_map` for this exact *mask_bool*
    (e.g. a settings search re-checking the same fixed mask many times
    across many candidates) can pass it through to skip a full-volume EDT
    it has already paid for. The caller is responsible for it actually
    matching *mask_bool*; nothing here re-validates that.
    """
    spacing = tuple(float(v) for v in voxel_size_zyx)
    local_radius = (
        local_radius_map if local_radius_map is not None else inscribed_radius_map(mask_bool, spacing)
    )
    discretisation_margin = float(np.linalg.norm(spacing))
    if source_bool.any():
        distance_to_source = distance_transform_edt(~source_bool, sampling=spacing)
    else:
        distance_to_source = np.full(mask_bool.shape, np.inf)
    return mask_bool & (distance_to_source <= local_radius + discretisation_margin)


def _missing_mask_components(
    explained: np.ndarray,
    mask_bool: np.ndarray,
    *,
    min_vessel_voxels: int,
) -> dict[str, Any]:
    """How many of *mask_bool*'s 26-connected components ("vessels") have no
    ``True`` voxel at all in *explained* (see :func:`_explained_by_local_radius`).

    This is the inverse question to the coverage-fraction checks above: those
    measure how much of the mask's *volume* is well-traced, which a whole
    small vessel going missing might barely move if it is a tiny fraction of
    total volume; this instead asks, per discrete vessel, "is any of it
    represented at all" -- catching a vessel dropped whole (e.g. by an
    over-aggressive small-object or connectivity filter) that a single
    blended percentage can hide.

    A component with fewer than *min_vessel_voxels* voxels is not a genuine
    vessel -- it is treated as segmentation noise and excluded from both the
    count and the check. Real segmented volumes routinely carry a handful of
    stray single-voxel thresholding specks alongside the actual vasculature;
    without this floor, a healthy run that correctly leaves that noise
    unskeletonised would otherwise report dozens of "missing vessels" that
    were never vessels to begin with.

    Shared by ``diagnose_vessels_missing_from_skeleton`` (this module) and
    ``graph.diagnostics.diagnose_vessels_missing_from_graph``, which differ
    only in what they pass as *explained*.

    The count alone overstates the loss: every component counts once, so a
    mask whose vasculature is one large component plus hundreds of specks
    reads as mostly missing (E14.5: 832 of 1,205 "vessels", 0.36% of the
    mask's volume). ``missing_volume_fraction`` -- the missing components'
    voxels over every mask voxel, specks below the floor included -- is
    reported beside it for that reason.
    """
    mask_voxel_count = int(mask_bool.sum())
    nothing_missing = {
        "vessel_count": 0,
        "missing_vessel_count": 0,
        "missing_vessel_voxel_counts": [],
        "explained_vessel_fraction": 1.0,
        "mask_voxel_count": mask_voxel_count,
        "missing_voxel_count": 0,
        "missing_volume_fraction": 0.0,
    }
    structure = generate_binary_structure(mask_bool.ndim, mask_bool.ndim)
    labeled, n_components = label(mask_bool, structure=structure)
    if n_components == 0:
        return nothing_missing

    sizes = np.bincount(labeled.ravel())
    vessel_ids = [i for i in range(1, n_components + 1) if sizes[i] >= min_vessel_voxels]
    if not vessel_ids:
        return nothing_missing

    # One vectorised pass over the whole volume rather than a fresh
    # full-volume `labeled == vessel_id` boolean array per component --
    # `ndi_sum` of a boolean array within each label is its True-voxel
    # count, so "no True voxel at all" is just "count == 0". Matters once
    # this is called from inside a settings-search sweep loop, not just
    # once per real pipeline run.
    explained_counts = ndi_sum(explained, labeled, index=vessel_ids)
    missing_voxel_counts = [
        int(sizes[vessel_id])
        for vessel_id, explained_count in zip(vessel_ids, explained_counts)
        if explained_count == 0
    ]
    vessel_count = len(vessel_ids)
    missing_vessel_count = len(missing_voxel_counts)
    missing_voxel_count = sum(missing_voxel_counts)
    return {
        "vessel_count": vessel_count,
        "missing_vessel_count": missing_vessel_count,
        "missing_vessel_voxel_counts": missing_voxel_counts,
        "explained_vessel_fraction": (vessel_count - missing_vessel_count) / vessel_count,
        "mask_voxel_count": mask_voxel_count,
        "missing_voxel_count": missing_voxel_count,
        "missing_volume_fraction": missing_voxel_count / mask_voxel_count,
    }


def _missing_volume_clause(report: dict[str, Any]) -> str:
    """How much of the mask the missing components are, for the two
    missing-vessel one-liners; empty when nothing is missing (or *report*
    predates the volume keys)."""
    counts = report.get("missing_vessel_voxel_counts") or []
    fraction = report.get("missing_volume_fraction")
    if not counts or fraction is None:
        return ""
    share = "<0.01%" if 0 < fraction < 0.0001 else f"{fraction:.2%}"
    largest = max(counts)
    return (
        f" Together they are {share} of the segmented volume; the largest is "
        f"{largest} voxel{'' if largest == 1 else 's'}."
    )


def diagnose_skeleton_mask_consistency(
    skeleton: np.ndarray,
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    local_radius_map: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Fraction of *mask* the skeleton actually runs through.

    A mask voxel counts as "explained" when the nearest skeleton voxel is no
    farther from it than that point's own inscribed radius, plus one voxel
    diagonal's worth of grid-discretisation slack (see below) -- the local
    vessel half-thickness, from the same EDT
    :func:`thick_vessels.inscribed_radius_map` uses for thickness gating. A
    skeleton passing further than that from part of a vessel's own interior
    is not running through that part of the mask. A whole region
    skeletonisation dropped, an off-centre ridge, or a vessel binarised too
    loosely for its centreline to be recovered cleanly would all show up
    here as a low fraction -- before that loss reaches the graph at all.

    The one-voxel-diagonal margin is not slack added to hide a real problem;
    it corrects a genuine miscalibration this module's own tests caught on a
    real fixture. Microvascular capillaries are frequently only 1-2 voxels
    wide, where the inscribed radius is itself close to its theoretical
    minimum of one voxel -- and a discretised medial axis running along one
    face of, say, a 2x2-voxel cross-section is, correctly, up to a full
    voxel diagonal from the section's opposite corner, even though there is
    no single-voxel-wide line that could do better. Comparing against the
    bare inscribed radius on such a fixture read a skeleton that visibly
    tracked the mask well as only 46% "explained"; adding one voxel's worth
    of margin (accounting for a half-voxel of placement slack on each of the
    skeleton's own discrete position and the mask's own EDT-based radius
    estimate) raised that to 93-95%, in line with what the skeleton actually
    looks like. Wider vessels are unaffected in relative terms, since the
    margin is a small, fixed addition next to their much larger radius.

    *mask* is read the same way skeletonisation itself reads it -- via
    :func:`io.load._to_binary_volume_for_skeletonization`, not a plain
    ``!= 0`` test. The image this compares against is whatever was loaded,
    not necessarily an already-clean 0/1 mask (a noisy or grayscale-ish
    input is thresholded internally before skeletonising it), so treating
    every nonzero voxel as foreground can read almost the entire volume as
    "mask" on such an input and make a perfectly healthy skeleton look like
    it explains almost none of it -- the same real fixture above has only
    1,230 of its 110,592 voxels genuinely foreground once thresholded this
    way, not the 105,594 a bare ``!= 0`` reads. (Imported locally: ``io.load``
    imports ``preprocessing.skeleton`` at module scope, so importing it back
    at this module's own top level would be circular.)

    *local_radius_map*, when given, is passed straight through to
    :func:`_explained_by_local_radius` instead of recomputing it from
    *mask* -- see that function's own docstring.
    """
    from haemolynx.io.load import _to_binary_volume_for_skeletonization

    mask_bool = _to_binary_volume_for_skeletonization(mask)
    mask_voxel_count = int(mask_bool.sum())
    if mask_voxel_count == 0:
        return {
            "mask_voxel_count": 0,
            "explained_voxel_count": 0,
            "coverage_fraction": 1.0,
        }

    skeleton_bool = np.asarray(skeleton, dtype=bool)
    explained = _explained_by_local_radius(
        skeleton_bool, mask_bool, voxel_size_zyx=voxel_size_zyx, local_radius_map=local_radius_map,
    )
    explained_voxel_count = int(explained.sum())
    return {
        "mask_voxel_count": mask_voxel_count,
        "explained_voxel_count": explained_voxel_count,
        "coverage_fraction": explained_voxel_count / mask_voxel_count,
    }


def format_skeleton_mask_consistency_report(report: dict[str, Any]) -> str:
    """A one-line summary of :func:`diagnose_skeleton_mask_consistency`."""
    return (
        "Skeleton/mask consistency: "
        f"{report.get('explained_voxel_count', 0)} of "
        f"{report.get('mask_voxel_count', 0)} segmented-image voxels are "
        f"within their own local radius (plus discretisation margin) of the "
        f"skeleton ({report.get('coverage_fraction', 1.0):.1%})."
    )


def diagnose_vessels_missing_from_skeleton(
    skeleton: np.ndarray,
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_vessel_voxels: int = 2,
    local_radius_map: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Whole segmented-image vessels the skeleton drops entirely.

    The inverse question to :func:`diagnose_skeleton_mask_consistency`'s
    coverage fraction: that measures how much of the mask's *volume* the
    skeleton runs through, which can stay comfortably high even while an
    entire small vessel is completely unrepresented, as long as it is a
    small enough slice of total volume. This instead treats each connected
    component of the mask as one candidate vessel and asks whether the
    skeleton explains *any* of it at all -- see
    :func:`_missing_mask_components` for why that per-vessel framing, and
    the noise-vessel size floor, matter.

    *mask* is read via the same canonical binarisation the other checks in
    this family use (see :func:`diagnose_skeleton_mask_consistency`).

    *local_radius_map*, when given, is passed straight through to
    :func:`_explained_by_local_radius` instead of recomputing it from
    *mask* -- see that function's own docstring.
    """
    from haemolynx.io.load import _to_binary_volume_for_skeletonization

    mask_bool = _to_binary_volume_for_skeletonization(mask)
    skeleton_bool = np.asarray(skeleton, dtype=bool)
    explained = _explained_by_local_radius(
        skeleton_bool, mask_bool, voxel_size_zyx=voxel_size_zyx, local_radius_map=local_radius_map,
    )
    return _missing_mask_components(
        explained, mask_bool, min_vessel_voxels=min_vessel_voxels
    )


def diagnose_skeleton_against_mask(
    skeleton: np.ndarray,
    mask: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_vessel_voxels: int = 2,
    local_radius_map: Optional[np.ndarray] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(diagnose_skeleton_mask_consistency(...),
    diagnose_vessels_missing_from_skeleton(...))`` for the same pair, from
    one binarisation, one local-radius map and one distance transform of
    the skeleton instead of two of each -- the same two reports.
    """
    from haemolynx.io.load import _to_binary_volume_for_skeletonization

    mask_bool = _to_binary_volume_for_skeletonization(mask)
    mask_voxel_count = int(mask_bool.sum())
    if mask_voxel_count == 0:
        consistency = {"mask_voxel_count": 0, "explained_voxel_count": 0, "coverage_fraction": 1.0}
        return consistency, _missing_mask_components(
            mask_bool, mask_bool, min_vessel_voxels=min_vessel_voxels
        )
    explained = _explained_by_local_radius(
        np.asarray(skeleton, dtype=bool),
        mask_bool,
        voxel_size_zyx=voxel_size_zyx,
        local_radius_map=local_radius_map,
    )
    explained_voxel_count = int(explained.sum())
    consistency = {
        "mask_voxel_count": mask_voxel_count,
        "explained_voxel_count": explained_voxel_count,
        "coverage_fraction": explained_voxel_count / mask_voxel_count,
    }
    return consistency, _missing_mask_components(
        explained, mask_bool, min_vessel_voxels=min_vessel_voxels
    )


#: Where a stretch of centreline the thick gate drops lies
#: (:func:`diagnose_centreline_dropped_by_thick_gate`): inside the thick
#: region, in the shell round it the flake filter never thins, or beyond.
DROPPED_PLACES = ("thick_region", "shell", "beyond")


def _distance_to_region_um(points: np.ndarray, region: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Each voxel of *points* (indices): 0 inside *region*, else the distance
    (um) to the nearest voxel of its surface -- a tree over the surface, not
    a volume-sized distance map."""
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree

    region = np.asarray(region, dtype=bool)
    distance = np.zeros(len(points))
    outside = ~region[tuple(points.T)] if len(points) else np.zeros(0, dtype=bool)
    surface = np.argwhere(region & ~binary_erosion(region, border_value=1))
    if outside.any():
        if len(surface):
            distance[outside] = cKDTree(surface * spacing).query(points[outside] * spacing)[0]
        else:
            distance[outside] = np.inf
    return distance


def diagnose_centreline_dropped_by_thick_gate(
    skeleton: np.ndarray,
    mask: np.ndarray,
    thick_region: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    large_mask: Optional[np.ndarray] = None,
    reach_um: float = 3.0,
    shell_um: float = 3.0,
    min_piece_voxels: int = 10,
    plain_skeleton: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Centreline plain Lee would draw that the thickness-gated *skeleton* has
    not: what ``use_thick_vessel_skeletonisation`` drops.

    The gated skeleton draws the *thick_region* (the large-vessel masks under
    ``skeleton_thick_vessel_restrict_to_mask``, plus their wall) as one
    centreline tree, and never Lee-thins thin voxels within the flake filter
    of it. Plain Lee of the hole-filled *mask* (*plain_skeleton*, when the
    caller has it) is compared with it: its voxels further than *reach_um*
    from every *skeleton* voxel, in 26-connected pieces of at least
    *min_piece_voxels*, are the centreline dropped, reported by where they
    lie (:data:`DROPPED_PLACES`: inside the region, within *shell_um* of
    it, beyond) and, with *large_mask*, how much is inside the large-vessel
    masks. Lengths count each voxel as the step to its nearest neighbour in
    its piece.

    Read-only, and a census, not a verdict: on E14.5 Harvey's grid P
    (2026-10-18, ``PLAN_surface_coverage.md``) called 25 of 30 such pieces
    the large vessel's own lumen or wall, which is what the gate exists to
    drop. It costs a whole-volume Lee, so the pipeline does not run it;
    ``scripts/graph_artefacts.py --by-place`` does.
    """
    from scipy.spatial import cKDTree

    from .skeleton import fill_binary_holes, skeletonize_volume

    spacing = np.asarray(voxel_size_zyx, dtype=float)
    mask_bool = np.asarray(mask, dtype=bool)
    if plain_skeleton is None:
        plain_skeleton = skeletonize_volume(fill_binary_holes(mask_bool)) if mask_bool.any() else mask_bool
    plain = np.argwhere(np.asarray(plain_skeleton, dtype=bool))
    gated = np.argwhere(np.asarray(skeleton, dtype=bool))
    report: dict[str, Any] = {
        "plain_voxels": len(plain),
        "dropped_voxels": 0,
        "dropped_pieces": 0,
        "dropped_um": 0.0,
        "by_place": {place: {"voxels": 0, "um": 0.0} for place in DROPPED_PLACES},
        "inside_large_um": None if large_mask is None else 0.0,
        "pieces": [],
    }
    if not len(plain):
        return report
    if len(gated):
        far = cKDTree(gated * spacing).query(plain * spacing, distance_upper_bound=float(reach_um) + 1e-9)[0]
        far = ~np.isfinite(far)
    else:
        far = np.ones(len(plain), dtype=bool)
    candidates = plain[far]
    if not len(candidates):
        return report
    shape = mask_bool.shape
    volume = np.zeros(shape, dtype=bool)
    volume[tuple(candidates.T)] = True
    labelled, _count = label(volume, structure=generate_binary_structure(3, 3))
    ids = labelled[tuple(candidates.T)]
    sizes = np.bincount(ids)
    kept = sizes[ids] >= int(min_piece_voxels)
    voxels, ids = candidates[kept], ids[kept]
    if not len(voxels):
        return report
    points = voxels * spacing
    # The step to the nearest other dropped voxel is within the same piece:
    # a nearer voxel of another piece would be 26-adjacent, so the same piece.
    step = cKDTree(points).query(points, k=2)[0][:, 1]
    to_region = _distance_to_region_um(voxels, thick_region, spacing)
    place = np.where(to_region == 0, 0, np.where(to_region <= float(shell_um), 1, 2))
    inside_large = None if large_mask is None else np.asarray(large_mask, dtype=bool)[tuple(voxels.T)]
    for i, name in enumerate(DROPPED_PLACES):
        report["by_place"][name] = {"voxels": int(np.sum(place == i)), "um": float(step[place == i].sum())}
    if inside_large is not None:
        report["inside_large_um"] = float(step[inside_large].sum())
    pieces = []
    for piece_id in np.unique(ids):
        mine = ids == piece_id
        centre = voxels[mine][np.argmin(np.linalg.norm(points[mine] - points[mine].mean(axis=0), axis=1))]
        pieces.append({
            "voxels": int(mine.sum()),
            "um": float(step[mine].sum()),
            "centre_zyx_vox": tuple(int(c) for c in centre),
            **{f"{name}_um": float(step[mine & (place == i)].sum()) for i, name in enumerate(DROPPED_PLACES)},
            "inside_large_um": None if inside_large is None else float(step[mine & inside_large].sum()),
        })
    pieces.sort(key=lambda piece: -piece["um"])
    report.update(
        dropped_voxels=len(voxels), dropped_pieces=len(pieces), dropped_um=float(step.sum()), pieces=pieces,
    )
    return report


def format_centreline_dropped_by_thick_gate_report(report: dict[str, Any], *, shell_um: float = 3.0) -> str:
    """A one-line summary of :func:`diagnose_centreline_dropped_by_thick_gate`."""
    places = report.get("by_place") or {}

    def um(place: str) -> str:
        return f"{(places.get(place) or {}).get('um', 0.0):,.0f}"

    large = report.get("inside_large_um")
    return (
        "Centreline the thick gate drops: "
        f"{report.get('dropped_um', 0.0):,.0f} um of plain-Lee centreline in {report.get('dropped_pieces', 0)} "
        f"piece(s) has no skeleton near it -- {um('thick_region')} um inside the thick region, "
        f"{um('shell')} um within {shell_um:g} um of it, {um('beyond')} um beyond"
        + ("." if large is None else f"; {large:,.0f} um inside the large-vessel masks.")
    )


def format_vessels_missing_from_skeleton_report(report: dict[str, Any]) -> str:
    """A one-line summary of :func:`diagnose_vessels_missing_from_skeleton`."""
    return (
        "Vessels missing from skeleton: "
        f"{report.get('missing_vessel_count', 0)} of "
        f"{report.get('vessel_count', 0)} segmented-image vessels have no "
        f"skeleton voxel anywhere within their own local radius "
        f"({report.get('explained_vessel_fraction', 1.0):.1%} represented)."
        + _missing_volume_clause(report)
    )
