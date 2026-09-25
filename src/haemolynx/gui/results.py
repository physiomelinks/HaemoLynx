"""What a finished stage should put in the viewer, worked out without napari.

:mod:`haemolynx.gui.layers` is the input direction -- what a layer already open
means for a run. This is the output direction: what a stage's own return value
means for the viewer. Nothing here imports napari; a spec is data, and the
widget's only job is to hand it to ``viewer.add_*``.

Two things decide the shape of this module.

**Conversion is eager, on purpose.** Every stage after ``build_network`` writes
attributes onto the same graph object. A spec built later -- on the GUI thread,
after the run has moved on -- would show a later stage's numbers under an
earlier stage's name: wrong, and silently so. So each stage's spec is built the
moment that stage hands its output over, on whatever thread the run is on.

**Geometry is usually built once.** Later stages normally only write attributes,
so vessels/nodes can keep their geometry and swap ``features`` (which is what
makes "colour by flow" a column switch). The exception is
``assign_boundaries`` when ``cut_network_at_large_vessel_volumes`` runs: that
stage rewrites the graph, so vessels/nodes must be rebuilt — including the
empty-graph case, otherwise a pre-cut Vectors layer stays on screen.

Coordinates: node ``pos`` and edge ``voxels`` are physical microns already --
voxel indices multiplied by ``voxel_size_zyx`` when the graph was built -- while
``image``, ``skeleton`` and the masks are voxel-indexed arrays. So graph layers
take ``scale=(1, 1, 1)`` and array layers take ``scale=voxel_size_zyx``. Getting
that backwards is invisible on isotropic data and wrong on every real stack.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from haemolynx.visualization._helpers import (
    create_color_mapping,
    sort_branch_orders_numerically,
)
from haemolynx.pipeline.stages import TOPOLOGY_STEP
from haemolynx.visualization.geometry import edge_polyline

logger = logging.getLogger(__name__)

#: Prefix on every layer this module names, so a re-run can tell its own layers
#: from the user's and never overwrite theirs.
PREFIX = "HaemoLynx "

VESSELS = f"{PREFIX}vessels"
#: Per-segment tube Surface drawn over :data:`VESSELS` in the napari view.
#: View-only — not a ResultLayers spec; the widget builds it from Vectors data.
VESSEL_TUBES = f"{PREFIX}vessel tubes"
VESSEL_LABELS = f"{PREFIX}vessel labels"
#: Legacy name of the midpoint-circle hover layer. Hover now lives on the
#: vessels Vectors polyline; the widget drops a leftover layer of this name.
BRANCH_HOVER = f"{PREFIX}branch hover"
#: Mid-edge arrows coloured by |flow|; emitted from Export when toggled on.
FLOW_DIRECTION = f"{PREFIX}flow direction"
NODES = f"{PREFIX}nodes"
BOUNDARY_NODES = f"{PREFIX}boundary nodes"
PERICYTES = f"{PREFIX}pericytes"
IMAGE = f"{PREFIX}image"
SKELETON = f"{PREFIX}skeleton"
FWHM_RAW = f"{PREFIX}FWHM image"
FWHM_PROFILES = f"{PREFIX}FWHM profiles"
FWHM_WIDTHS = f"{PREFIX}FWHM measured widths"
#: The "Edit" window's in-progress "Add branch" path, drawn separately from
#: VESSELS since it is not part of the graph until the draft is committed.
EDIT_DRAFT = f"{PREFIX}edit draft"

#: Napari Points ``size`` (data pixels). Values match the original viewer style
#: on ``origin/main`` / the first GUI results commit. Branch hover used to be a
#: midpoint circle at 8.0, then 2.0; it is no longer a Points layer.
VESSEL_LABEL_POINT_SIZE = 2.0
NODE_POINT_SIZE = 3.0
BOUNDARY_NODE_POINT_SIZE = 6.0
PERICYTE_POINT_SIZE = 4.0
#: Picking rings on the Boundaries tab (distinct from snapped boundary nodes).
BOUNDARY_COORDINATE_POINT_SIZE = 8.0

#: Mask field on VesselNetwork -> the layer it becomes.
MASK_LAYERS = {
    "large_arteriole_mask": f"{PREFIX}large arteriole mask",
    "large_venule_mask": f"{PREFIX}large venule mask",
    "small_arteriole_mask": f"{PREFIX}small arteriole mask",
    "small_venule_mask": f"{PREFIX}small venule mask",
}

#: Translucent RGBA for each mask role in 3D volume rendering (widget builds
#: a two-stop colormap: transparent at 0, this colour at 1). Hex equivalents:
#: large arteriole dark red ``#8C0D0D``, small arteriole light red ``#FF7373``,
#: large venule dark green ``#0D661A``, small venule light green ``#73E673``.
MASK_COLOURS: dict[str, tuple[float, float, float, float]] = {
    "large_arteriole_mask": (0.55, 0.05, 0.05, 0.70),
    "small_arteriole_mask": (1.00, 0.45, 0.45, 0.55),
    "large_venule_mask": (0.05, 0.40, 0.10, 0.70),
    "small_venule_mask": (0.45, 0.90, 0.45, 0.55),
}

#: Shared napari Image options for vessel-mask volumes (plus per-role colour).
MASK_VOLUME_OPTIONS: dict[str, Any] = {
    "blending": "translucent",
    "opacity": 0.55,
    "rendering": "mip",
    "interpolation2d": "nearest",
    "interpolation3d": "nearest",
}

#: Neutral, role-free colour for the segmented input image when it turns out
#: to be a binary-ish mask (see ``binary_value_range``) -- distinct from
#: every ``MASK_COLOURS`` role so it never reads as one of them.
SEGMENTED_IMAGE_COLOUR: tuple[float, float, float, float] = (0.78, 0.78, 0.82, 0.6)

#: The segmented-image layer's own look once its data is binary-ish. Plain
#: "gray" (this pipeline's default for genuine grayscale) has no notion of
#: "background", so a dense, mostly space-filling vessel volume rendered
#: that way turns solid: MIP-family rendering shows the single brightest
#: sample along each ray regardless of depth, and on this data nearly every
#: ray hits foreground somewhere. ``rendering: "translucent"`` composites
#: each voxel's own opacity front-to-back instead of an any-hit silhouette,
#: so actual 3D structure reads rather than a flat outline; the two-stop
#: transparent-to-colour colormap this pairs with (built from ``mask_colour``,
#: the same mechanism vessel-mask overlays already use) is what makes the
#: background stop occluding whatever is drawn behind or around it.
#: ``blending: "translucent_no_depth"`` (not plain "translucent") turns off
#: GPU depth-testing for this layer specifically -- with depth-testing on,
#: a voxel whose alpha happens to be near-zero under the *current* colormap
#: can still write the depth buffer, so switching to a colormap with a
#: different alpha profile (or just a different contrast) can leave parts
#: of the volume behind it wrongly discarded rather than blended. No depth
#: test means every visible voxel always composites correctly regardless
#: of which colormap is applied.
BINARY_IMAGE_VOLUME_OPTIONS: dict[str, Any] = {
    "blending": "translucent_no_depth",
    "rendering": "translucent",
    "interpolation2d": "nearest",
    "interpolation3d": "nearest",
}


def _on_disk(data: Any) -> bool:
    """A disk-backed volume -- what the low-RAM option loads -- is read a slab
    at a time here, so displaying it never builds a whole-volume temporary.
    Every function below gives the same answer either way."""
    return isinstance(data, np.memmap)


def _display_owned(volume: Any) -> Any:
    """A disk-backed volume made only to be displayed goes when the viewer
    drops it."""
    if isinstance(volume, np.memmap):
        from haemolynx.preprocessing.memmap_support import delete_when_unmapped

        delete_when_unmapped(volume)
    return volume


def _slabs(volume: np.ndarray):
    from haemolynx.preprocessing.memmap_support import LOW_MEMORY_BLOCK_VOXELS

    slice_voxels = max(1, int(np.prod(volume.shape[1:], dtype=np.int64)))
    step = max(1, LOW_MEMORY_BLOCK_VOXELS // slice_voxels)
    for start in range(0, volume.shape[0], step):
        yield np.asarray(volume[start:start + step])


def binary_value_range(data: Any) -> tuple[float, float] | None:
    """``(low, high)`` if *data* takes on exactly two distinct values.

    A segmented mask ships as any of 0/1, 1/2 or 0/255 -- never assume which.
    Returns ``None`` for a blank array (nothing to threshold) or one with
    more than two distinct values (genuine grayscale), so a caller can fall
    back to treating it as a continuous image.
    """
    if _on_disk(data) and data.ndim >= 1 and data.size:
        low = float(min(slab.min() for slab in _slabs(data)))
        high = float(max(slab.max() for slab in _slabs(data)))
        if low == high:
            return None
        if not all(bool(np.all((slab == low) | (slab == high))) for slab in _slabs(data)):
            return None
        return (low, high)
    array = np.asarray(data)
    if array.size == 0:
        return None
    low = float(array.min())
    high = float(array.max())
    if low == high:
        return None
    if not bool(np.all((array == low) | (array == high))):
        return None
    return (low, high)


def binarize_for_display(data: Any) -> np.ndarray | None:
    """The already-binarised array to render *data* as a transparent-
    background mask, for a segmented image ``binary_value_range`` misses --
    or ``None`` when *data* does not look like a segmentation mask at all.

    A real segmented image is not always exactly two-valued the way
    ``binary_value_range`` requires: an ilastik "Simple Segmentation"
    export, for instance, can carry a handful of unclassified-border pixels
    at 0 alongside its 1/2 class labels, three distinct values where
    ``binary_value_range`` sees "genuine grayscale" and falls back to a
    flat, opaque colour block -- the segmented image rendering an
    apparently solid box in 3D even though the earlier binary-image fix is
    otherwise working exactly as designed.

    Confidently a mask, not real grayscale: an integer array with 3 or 4
    distinct values (matching ``io.load._to_binary_volume_for_skeletonization``'s
    own small-label-set heuristic -- beyond 4 it falls back to an
    Otsu-style threshold instead, which this deliberately does not mirror)
    where one value covers more than half of all voxels. A tiny grayscale
    sample can easily land on 3-4 distinct integer values too, but real
    image content essentially never splits into a handful of values with no
    majority at all the way a background-dominated segmentation mask does
    -- a real production volume with 89.8% background is comfortably past
    this bar, while a synthetic four-pixel grayscale fixture with one of
    each value (25% each) is comfortably short of it.

    Reuses the pipeline's own canonical binarisation for the actual
    foreground split once this check passes, so what is displayed always
    matches what the pipeline analyses.
    """
    on_disk = _on_disk(data)
    array = data if on_disk else np.asarray(data)
    if array.size == 0 or not np.issubdtype(array.dtype, np.integer):
        return None
    from haemolynx.io.load import _to_binary_volume_for_skeletonization, _unique_with_counts

    if on_disk:
        _values, counts = _unique_with_counts(array, use_memmap=True)
    else:
        _values, counts = np.unique(array, return_counts=True)
    if _values.size not in (3, 4):
        return None
    if float(counts.max()) / float(array.size) <= 0.5:
        return None
    return _display_owned(_to_binary_volume_for_skeletonization(array, use_memmap=on_disk))


#: The fixed names this module emits -- one set of layers per run, whatever the
#: settings say. It is *not* the whole set any more: a perturbation's layers are
#: named after the perturbation, so they cannot be enumerated ahead of a run.
#: :func:`is_ours_name` is the question worth asking of a name, and "clear ours"
#: asks the layer itself, through the `OURS` metadata tag it was added with.
LAYER_NAMES = frozenset(
    {
        VESSELS,
        VESSEL_TUBES,
        VESSEL_LABELS,
        BRANCH_HOVER,
        FLOW_DIRECTION,
        NODES,
        BOUNDARY_NODES,
        PERICYTES,
        IMAGE,
        SKELETON,
        FWHM_RAW,
        FWHM_PROFILES,
        FWHM_WIDTHS,
    }
    | set(MASK_LAYERS.values())
)

#: Voxel-indexed arrays and boundary-picking layers are not graph geometry.
_Z_FILTER_EXCLUDE = frozenset({IMAGE, SKELETON, FWHM_RAW, *MASK_LAYERS.values()})


def is_ours_name(name: str) -> bool:
    """Whether *name* is a name this module would give a layer.

    A predicate rather than a set, because `run_perturbations` names its
    layers after the perturbations a config asked for. Naming is still all it
    tells you: a layer is *ours* by the metadata it was added with, which is
    what stops "clear layers" removing a user's layer that happens to be
    called the same thing.
    """
    return str(name).startswith(PREFIX)


def perturbation_layer_names(name: str) -> tuple[str, str]:
    """The vessels and nodes layers one perturbation gets, in that order."""
    return (f"{PREFIX}{name} vessels", f"{PREFIX}{name} nodes")


def perturbation_flow_direction_layer_name(name: str) -> str:
    """The flow-direction arrows layer of one (single re-solve) perturbation."""
    return f"{PREFIX}{name} flow direction"


def image_z_extent_um(
    voxel_size_zyx: Sequence[float], image_shape_z: int
) -> float:
    """Physical Z span of the loaded image stack, in microns."""
    return float(voxel_size_zyx[0]) * int(image_shape_z)


def is_z_depth_filtered_layer(name: str, kind: str) -> bool:
    """Whether the view-only Z depth filter clips graph geometry on *layer*.

    Image and labels volumes are clipped too, but they use the
    ``z_window_full`` cache and :func:`clip_volume_to_z` rather than this
    graph-geometry predicate. Boundary-picking layers stay unfiltered so a
    picker cannot hide the points it is meant to edit.
    """
    if kind not in ("vectors", "points"):
        return False
    if not is_ours_name(name):
        return False
    if name in _Z_FILTER_EXCLUDE:
        return False
    # Boundary-picking layers (``boundary_picking.py``) share the prefix.
    if " BC " in name:
        return False
    return True


def is_z_depth_windowed_volume_layer(kind: str) -> bool:
    """Whether the view-only Z depth filter clips this layer kind by caching
    the full volume and restoring a clipped copy (image/labels), rather than
    filtering rows the way :func:`is_z_depth_filtered_layer` does for graph
    geometry (vectors/points)."""
    return kind in ("image", "labels")


def z_window_is_full(z_min: float, z_max: float, z_extent: float | None) -> bool:
    """Whether ``[z_min, z_max]`` covers the full Z extent, or is degenerate.

    A window narrower than 1e-6 (both handles on the same value, e.g. before
    the user has moved the slider, or a collapsed programmatic ``setValue``)
    is treated the same as "full": there is nothing meaningful to filter to,
    so callers fall back to showing everything rather than clipping to an
    effectively empty band.
    """
    if z_max <= z_min + 1e-6:
        return True
    if z_extent is None:
        return False
    return (
        z_min <= 0.0
        and z_max >= z_extent - max(1e-6, abs(z_extent) * 1e-9)
    )


def z_slice_window(
    n_z: int, voxel_size_z: float, z_min: float, z_max: float
) -> tuple[int, int]:
    """Inclusive slice indices whose origins lie in ``[z_min, z_max]`` µm.

    Slice ``i`` is anchored at ``i * voxel_size_z``, matching
    :func:`image_z_extent_um` (extent ``n_z * voxel_size_z``).
    """
    n_z = int(n_z)
    if n_z <= 0:
        return 0, 0
    dz = float(voxel_size_z) if voxel_size_z else 1.0
    starts = np.arange(n_z, dtype=float) * dz
    keep = (starts >= z_min - 1e-9) & (starts <= z_max + 1e-9)
    if not np.any(keep):
        idx = int(np.clip(np.round(0.5 * (z_min + z_max) / dz), 0, n_z - 1))
        return idx, idx
    indices = np.flatnonzero(keep)
    return int(indices[0]), int(indices[-1])


def clip_volume_to_z(
    volume: np.ndarray,
    voxel_size_z: float,
    z_min: float,
    z_max: float,
    *,
    z_extent: float | None = None,
) -> np.ndarray:
    """Zero slices outside ``[z_min, z_max]`` µm, keeping the 3D shape.

    Slices inside the window keep their original values (no MIP). A window
    covering the full extent returns *volume* unchanged. ``z_max < z_min``
    returns zeros. Display-only — the pipeline must keep the original stack.
    """
    on_disk = _on_disk(volume)
    volume = volume if on_disk else np.asarray(volume)
    if volume.ndim < 3:
        return volume
    if on_disk:
        from haemolynx.preprocessing.memmap_support import new_memmap_array

        zeros = lambda: _display_owned(new_memmap_array(volume.shape, volume.dtype))  # noqa: E731
    else:
        zeros = lambda: np.zeros_like(volume)  # noqa: E731
    if z_max < z_min:
        return zeros()
    n_z = int(volume.shape[0])
    dz = float(voxel_size_z) if voxel_size_z else 1.0
    extent = float(z_extent) if z_extent is not None else dz * n_z
    if z_window_is_full(z_min, z_max, extent):
        return volume
    start, stop = z_slice_window(n_z, dz, z_min, z_max)
    out = zeros()
    out[start : stop + 1] = volume[start : stop + 1]
    return out


def _z_in_range(z: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    return (z >= z_min) & (z <= z_max)


def filter_vectors_by_z(
    data: np.ndarray,
    features: Mapping[str, np.ndarray],
    z_min: float,
    z_max: float,
    *,
    segment_owner: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
    """Keep vector segments whose origin Z lies in ``[z_min, z_max]``."""
    data = np.asarray(data)
    if len(data) == 0:
        return data, dict(features), segment_owner
    keep = _z_in_range(data[:, 0, 0], z_min, z_max)
    filtered = {
        name: np.asarray(values)[keep] for name, values in features.items()
    }
    owner = np.asarray(segment_owner)[keep] if segment_owner is not None else None
    return data[keep], filtered, owner


def filter_points_by_z(
    data: np.ndarray,
    features: Mapping[str, np.ndarray],
    z_min: float,
    z_max: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Keep points whose position Z lies in ``[z_min, z_max]``."""
    data = np.asarray(data)
    if len(data) == 0:
        return data, dict(features)
    keep = _z_in_range(data[:, 0], z_min, z_max)
    filtered = {
        name: np.asarray(values)[keep] for name, values in features.items()
    }
    return data[keep], filtered


#: Per-edge columns, and the stage that first writes each. A column is offered
#: for colouring only once the stage that fills it has run, so the dropdown
#: never lists a quantity that would come back all-NaN.
EDGE_COLUMNS: dict[str, str] = {
    "length": "build_network",
    "segment_id": "build_network",
    "mask_vessel_type": "assign_boundaries",
    "branch_order": "assign_diameters",
    "diameter_um": "assign_diameters",
    "diameter_source": "assign_diameters",
    "fwhm_diameter_um": "assign_diameters",
    "fwhm_status": "assign_diameters",
    "pericyte_count_assigned": "assign_diameters",
    "resistance": "build_haemodynamic_model",
    "conductance": "build_haemodynamic_model",
    "pressure_u": "solve",
    "pressure_v": "solve",
    "pressure_drop": "solve",
    "flow_signed": "solve",
    "flow_abs": "solve",
}

#: Optional edge columns gated by run settings (see :func:`edge_columns_for_settings`).
OPTIONAL_EDGE_COLUMNS: dict[str, str] = {
    "flow_abs_log10": "solve",
    "flow_dir_z": "solve",
    "flow_dir_y": "solve",
    "flow_dir_x": "solve",
    "flow_heading_deg": "solve",
    "flow_dir_rgb": "solve",
    "edt_diameter_um": "assign_diameters",
    "fwhm_edt_disagreement_ratio": "assign_diameters",
    "fwhm_low_confidence_vs_edt": "assign_diameters",
    # Only present when haematocrit_model is distributed_iterative --
    # available_edge_columns already drops any column with no real values,
    # so this is silently absent from the dropdown otherwise.
    "discharge_haematocrit": "solve",
    # Written by graph.assign_vascular_communities when
    # compute_vascular_communities is on; empty on every edge otherwise,
    # exactly like fwhm_status when FWHM measurement never ran.
    "vascular_community": "export_results",
    # Written by statistics.inlet_outlet_routes when statistics_bottlenecks /
    # statistics_shunts ran with inlet and outlet nodes; absent otherwise,
    # so available_edge_columns keeps them out of the dropdown.
    "bottleneck_route_share": "export_results",
    "bottleneck_min_cut": "export_results",
    "shunt_route_ratio": "export_results",
    "shunt": "export_results",
    # The other annotating network analyses (see statistics.comprehensive).
    "occlusion_flow_loss": "export_results",
    "occlusion_hypoperfused_length_um": "export_results",
    "current_flow_share": "export_results",
    "arteriolar_territory": "export_results",
    "venular_territory": "export_results",
    "watershed": "export_results",
    "arteriolar_watershed_margin": "export_results",
    "transit_time_s": "export_results",
    "arrival_time_s": "export_results",
    "loop_length_um": "export_results",
    "strahler_order": "export_results",
    "fiedler_value": "export_results",
    "fiedler_side": "export_results",
    # Written onto a capillary-block perturbation's own graph (see
    # haemodynamics.capillary_block), so they only ever appear on that
    # perturbation's vessels layer.
    "capillary_block": "run_perturbations",
    "flow_change_vs_baseline": "run_perturbations",
}

#: Edge columns that scale with the flow itself, so every sweep point --
#: even a pressure-only one -- has its own values.
SWEEP_FLOW_MAGNITUDE_COLUMNS = frozenset({"transit_time_s", "arrival_time_s"})

#: Edge columns a dilation or geometry sweep point changes.
SWEEP_GEOMETRY_DEPENDENT_COLUMNS = frozenset({
    "diameter_um",
    "fwhm_diameter_um",
    "edt_diameter_um",
    "fwhm_edt_disagreement_ratio",
    "fwhm_low_confidence_vs_edt",
    "resistance",
    "conductance",
    "discharge_haematocrit",
})

#: Derived flow columns always offered on vessel layers once flows exist.
FLOW_DERIVED_EDGE_COLUMNS: frozenset[str] = frozenset(
    {
        "flow_abs_log10",
        "flow_dir_z",
        "flow_dir_y",
        "flow_dir_x",
        "flow_heading_deg",
        "flow_dir_rgb",
    }
)


def edge_columns_for_settings(
    settings: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Declared edge columns, including derived flow columns for colouring.

    ``flow_direction_colouring`` (default on) is the one setting this
    actually reads: when off, the raw ``flow_dir_z``/``flow_dir_y``/
    ``flow_dir_x`` axis components are dropped from the declared columns
    entirely, so they never appear as a colour-by option on any layer --
    not just left unfilled, since a declared-but-NaN column would still show
    up in the dropdown (see this module's own notes on why every column is
    declared up front). ``flow_dir_rgb``/``flow_heading_deg`` are derived,
    composite encodings, not "axis components", and stay regardless.
    """
    columns = dict(EDGE_COLUMNS)
    columns.update(OPTIONAL_EDGE_COLUMNS)
    if settings is not None and not settings.get("flow_direction_colouring", True):
        for name in FLOW_DIR_COLUMNS:
            columns.pop(name, None)
    return columns


def sweep_direction_columns(
    directions: tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]],
    signed_flow: np.ndarray,
) -> dict[str, np.ndarray]:
    """Flow-direction columns for one sweep grid point.

    *directions* is each drawable edge's columns as if it flowed ``u -> v``
    and as if ``v -> u`` (``edge_flow_direction_columns`` with a sign
    override); *signed_flow* is that point's ``flow_signed`` per drawable
    edge. Each edge takes the column for the way its flow actually runs --
    exactly what the columns would be if the network were solved at that
    point -- and NaN where it carries none.
    """
    forward, backward = directions
    signed = np.asarray(signed_flow, dtype=float)
    positive = np.isfinite(signed) & (signed > 0)
    negative = np.isfinite(signed) & (signed < 0)
    columns = {}
    for name, ahead in forward.items():
        ahead = np.asarray(ahead, dtype=float)
        behind = np.asarray(backward[name], dtype=float)
        if name == FLOW_DIR_RGB_COLUMN:
            columns[name] = np.zeros_like(ahead)  # a sentinel column, not a value
            continue
        columns[name] = np.where(positive, ahead, np.where(negative, behind, np.nan))
    return columns


def _enrich_flow_colour_columns(
    columns: dict[str, np.ndarray], graph: Any, *, include_axis_components: bool = True
) -> None:
    """Add log10 and direction columns whenever the graph carries finite flows.

    *include_axis_components* gates only the raw ``flow_dir_z``/``flow_dir_y``/
    ``flow_dir_x`` columns (the ``flow_direction_colouring`` setting's own
    "axis components" -- see its help text); ``flow_dir_rgb``/
    ``flow_heading_deg`` are derived, composite encodings and always included.
    """
    flow_abs = columns.get("flow_abs")
    if flow_abs is None:
        return
    values = np.asarray(flow_abs, dtype=float)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return
    from haemolynx.haemodynamics.resistance import flow_abs_log10_value
    from haemolynx.visualization.flow_direction import edge_flow_direction_columns

    columns["flow_abs_log10"] = np.asarray(
        [flow_abs_log10_value(v) if np.isfinite(v) else np.nan for v in values],
        dtype=float,
    )
    direction = edge_flow_direction_columns(graph)
    if not include_axis_components:
        direction = {k: v for k, v in direction.items() if k not in FLOW_DIR_COLUMNS}
    for name, array in direction.items():
        if len(array) == len(values):
            columns[name] = array


#: Columns holding text rather than numbers; a missing one is "" not NaN.
TEXT_COLUMNS = frozenset(
    {
        "branch_order",
        "mask_vessel_type",
        "diameter_source",
        "fwhm_status",
        "vascular_community",
        "bottleneck_min_cut",
        "shunt",
        "arteriolar_territory",
        "venular_territory",
        "watershed",
        "fiedler_side",
        "capillary_block",
    }
)

#: What each stage colours the vessels by once it has run, unless the user has
#: chosen otherwise. `flow_abs`, not `flow_signed`: the sign follows the order
#: the MultiGraph happens to store an edge in, so a signed colouring shows an
#: arbitrary pattern that reads as a physics bug.
DEFAULT_VESSEL_COLOUR = {
    "build_network": "segment_id",
    "assign_diameters": "diameter_um",
    "build_haemodynamic_model": "resistance",
    "solve": "flow_abs",
}

#: What a perturbation's own vessels are coloured by. The same quantity the
#: solved baseline is, so switching a perturbation on and off compares like
#: with like -- which is the only thing these layers are for.
PERTURBATION_COLOUR = "flow_abs"


@dataclass(frozen=True)
class LayerSpec:
    """One layer to add or update, described without napari."""

    kind: str  # image | labels | points | vectors | shapes
    name: str
    data: Any
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    features: Mapping[str, np.ndarray] = field(default_factory=dict)
    colour_by: str | None = None
    #: "continuous" drives a colormap, "categorical" a colour cycle,
    #: "direct" an RGB array (the 3D flow-direction map).
    colour_kind: str = "none"
    colour_cycle: tuple[tuple[str, tuple[float, float, float, float]], ...] = ()
    contrast_limits: tuple[float, float] | None = None
    visible: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)
    #: Compact sweep grid for slider-backed flow layers; not passed to napari.
    sweep: Any | None = None
    #: Per-segment owner index into drawable-edge columns (Vectors only).
    segment_owner: Any | None = None
    #: Drawable-edge indices into the full ``edges(keys=True)`` flow arrays.
    sweep_edge_index: Any | None = None
    #: ``(forward, backward)`` flow-direction columns per drawable edge, for
    #: :func:`sweep_direction_columns` to pick from at each grid point.
    sweep_directions: Any | None = None


@dataclass(frozen=True)
class StageLayers:
    """What one finished stage means for the viewer."""

    stage: str
    title: str
    layers: tuple[LayerSpec, ...] = ()
    #: Layers already there that only change how they are coloured.
    recolour: tuple[tuple[str, str], ...] = ()
    #: One line for the report box.
    note: str = ""
    #: 3 when the first geometry arrives, so a paths layer is not hidden by the
    #: 2D slice; None leaves the viewer's own setting alone.
    ndisplay: int | None = None


def vessel_mask_volume_layers(
    masks: Any,
    *,
    voxel_size_zyx: Sequence[float],
    visible: bool = True,
) -> tuple[LayerSpec, ...]:
    """3D image-volume specs for vessel masks that are actually present.

    *masks* is anything with the four VesselNetwork attributes (a real
    network, a ``SimpleNamespace``, or a mapping). A missing or ``None``
    attribute produces no layer -- disabled / unused masks stay out of the
    viewer. Arrays are voxel-indexed, so ``scale`` is ``voxel_size_zyx``.
    """
    scale = tuple(float(v) for v in voxel_size_zyx)
    if len(scale) != 3:
        raise ValueError(
            f"voxel_size_zyx must be three floats (z, y, x); got {voxel_size_zyx!r}."
        )
    layers: list[LayerSpec] = []
    for attribute, name in MASK_LAYERS.items():
        if isinstance(masks, Mapping):
            mask = masks.get(attribute)
        else:
            mask = getattr(masks, attribute, None)
        if mask is None:
            continue
        colour = MASK_COLOURS[attribute]
        layers.append(
            LayerSpec(
                kind="image",
                name=name,
                data=np.asarray(mask, dtype=np.float32),
                scale=scale,  # type: ignore[arg-type]
                contrast_limits=(0.0, 1.0),
                visible=visible,
                options={**MASK_VOLUME_OPTIONS, "mask_colour": colour},
            )
        )
    return tuple(layers)


# --- reading a graph ---------------------------------------------------------


def edge_polylines(graph: Any) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Every edge's polyline, plus the columns that identify it.

    Edges the geometry cannot place -- no voxels and no node positions -- are
    dropped rather than raising: one unplaceable vessel should not cost the
    other thousand their layer.
    """
    paths: list[np.ndarray] = []
    edge_index: list[int] = []
    us: list[Any] = []
    vs: list[Any] = []
    keys: list[int] = []

    for index, (u, v, key, data) in enumerate(_iter_edges(graph)):
        try:
            paths.append(edge_polyline(graph, u, v, data))
        except ValueError:
            continue
        edge_index.append(index)
        us.append(u)
        vs.append(v)
        keys.append(key)

    identity = {
        "edge_index": np.asarray(edge_index, dtype=int),
        "u": np.asarray(us),
        "v": np.asarray(vs),
        "key": np.asarray(keys, dtype=int),
    }
    return paths, identity


def _iter_edges(graph: Any) -> Iterable[tuple[Any, Any, int, Mapping[str, Any]]]:
    """(u, v, key, data) whether or not this is a MultiGraph."""
    if getattr(graph, "is_multigraph", lambda: False)():
        return graph.edges(keys=True, data=True)
    return ((u, v, 0, data) for u, v, data in graph.edges(data=True))


def polylines_to_vectors(
    paths: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Polylines as (M, 2, 3) origin+direction segments, and which path each came from.

    A Vectors layer draws every segment separately, so a per-edge value has to
    be repeated across that edge's segments -- the second array is the index to
    repeat by.
    """
    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    owner: list[int] = []
    for index, path in enumerate(paths):
        points = np.asarray(path, dtype=float)
        if len(points) < 2:
            continue
        starts = points[:-1]
        origins.append(starts)
        directions.append(points[1:] - starts)
        owner.extend([index] * len(starts))

    if not origins:
        return np.empty((0, 2, 3), dtype=float), np.empty(0, dtype=int)

    vectors = np.stack(
        [np.concatenate(origins, axis=0), np.concatenate(directions, axis=0)], axis=1
    )
    return vectors, np.asarray(owner, dtype=int)


def fwhm_profile_polylines(graph: Any) -> list[np.ndarray]:
    """The line each accepted FWHM sample was measured along.

    About ``min_total_extent_multiplier`` (3) times the vessel's width -- a
    half-maximum is only defined against background on both sides -- unless
    it ran into another vessel. A line no wider than the vessel is a
    measurement that never saw background. For the width itself, see
    :func:`fwhm_measured_polylines`.
    """
    return _stored_polylines(graph, "fwhm_profile_lines_phys")


def fwhm_measured_polylines(graph: Any) -> list[np.ndarray]:
    """One line per accepted FWHM sample, exactly as long as the diameter it
    measured and centred on the fitted centre: laid over the vessel, it
    should span it wall to wall."""
    return _stored_polylines(graph, "fwhm_measured_lines_phys")


def _stored_polylines(graph: Any, attribute: str) -> list[np.ndarray]:
    lines: list[np.ndarray] = []
    for _u, _v, _key, data in _iter_edges(graph):
        for line in data.get(attribute) or ():
            points = np.asarray(line, dtype=float)
            if points.ndim != 2 or len(points) < 2:
                continue
            lines.append(points[:, :3])
    return lines


def edge_features(graph: Any, names: Iterable[str]) -> dict[str, np.ndarray]:
    """One column per name, in edge order, for the edges that can be drawn.

    A value a stage has not written yet comes back NaN, or "" for text. The
    flow columns are sparse even after a solve -- `set_edge_flows` skips an edge
    with no conductance -- so a consumer must use nan-aware limits.
    """
    wanted = list(names)
    columns: dict[str, list[Any]] = {name: [] for name in wanted}

    for u, v, _key, data in _iter_edges(graph):
        try:
            edge_polyline(graph, u, v, data)
        except ValueError:
            continue
        for name in wanted:
            value = data.get(name)
            if name in TEXT_COLUMNS:
                columns[name].append("" if value is None else str(value))
            else:
                columns[name].append(np.nan if value is None else float(value))

    return {
        name: np.asarray(values, dtype=object if name in TEXT_COLUMNS else float)
        for name, values in columns.items()
    }


def available_edge_columns(
    graph: Any, settings: Mapping[str, Any] | None = None
) -> list[str]:
    """The columns this graph actually carries a value for, in declared order."""
    present: list[str] = []
    for name in edge_columns_for_settings(settings):
        for _u, _v, _key, data in _iter_edges(graph):
            if data.get(name) is not None:
                present.append(name)
                break
    return present


def node_points(
    graph: Any, node_ids: Sequence[Any] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Positions and ids for the nodes that have a position."""
    ids = list(graph.nodes) if node_ids is None else list(node_ids)
    points: list[np.ndarray] = []
    kept: list[Any] = []
    for node_id in ids:
        pos = graph.nodes[node_id].get("pos") if node_id in graph.nodes else None
        if pos is None:
            continue
        points.append(np.asarray(pos, dtype=float)[:3])
        kept.append(node_id)
    if not points:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=object)
    return np.stack(points), np.asarray(kept, dtype=object)


def colour_cycle_for(
    values: Iterable[Any],
) -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
    """Colours for a text column, matching the plots the same run writes."""
    labels = sorted({str(value) for value in values if str(value)})
    if not labels:
        return ()
    ordered = sort_branch_orders_numerically(labels)
    mapping = create_color_mapping(ordered)
    return tuple((label, tuple(float(c) for c in mapping[label])) for label in ordered)


def pericyte_points(graph: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Where the constrictions actually are, from each edge's own centres.

    `constriction.py` records `pericyte_centers_um` per edge -- arclengths along
    that edge, in microns -- so the points are read back by interpolating along
    the polyline. `visualization.derive_pericyte_points_from_graph` does not use
    them: it re-derives positions periodically from the spacing settings, which
    is right for the periodic strategy and wrong for the mask one, where the
    sites came from a segmented image. Here we want where they are.
    """
    from haemolynx.geometry import cumulative_lengths
    from haemolynx.visualization.vtk_io import _interpolate_at_length

    points: list[np.ndarray] = []
    edge_index: list[int] = []
    branch_order: list[str] = []
    arclength: list[float] = []

    for index, (u, v, _key, data) in enumerate(_iter_edges(graph)):
        centres = data.get("pericyte_centers_um")
        if not centres:
            continue
        try:
            path = edge_polyline(graph, u, v, data)
        except ValueError:
            continue
        cumlen = cumulative_lengths(path)
        for centre in centres:
            points.append(_interpolate_at_length(path, cumlen, float(centre)))
            edge_index.append(index)
            branch_order.append(str(data.get("branch_order", "")))
            arclength.append(float(centre))

    if not points:
        return np.empty((0, 3), dtype=float), {}
    return np.stack(points), {
        "edge_index": np.asarray(edge_index, dtype=int),
        "branch_order": np.asarray(branch_order, dtype=object),
        "arc_length_um": np.asarray(arclength, dtype=float),
    }


def midpoints_of(paths: Sequence[np.ndarray]) -> np.ndarray:
    """The middle of each polyline, for the hidden vessel-label layer."""
    if not paths:
        return np.empty((0, 3), dtype=float)
    return np.stack([np.asarray(path, dtype=float).mean(axis=0) for path in paths])


def _branch_hover_columns(
    graph: Any, edge_index: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Per-vector tooltip columns plus option keys the widget stashes.

    *edge_index* selects drawable edges (the Vectors ``owner`` array, or the
    drawable-edge index of each flow-direction arrow). Empty when there is
    nothing to hover.
    """
    from haemolynx.gui.branch_hover import hover_features_for_segments

    features, available, selected = hover_features_for_segments(graph, edge_index)
    if not features:
        return {}, {}
    return features, {
        "branch_hover_available": available,
        "branch_hover_selected": selected,
    }


def _flow_direction_drawable_edge_index(graph: Any) -> np.ndarray:
    """Drawable-edge indices of edges that receive a flow-direction arrow.

    Same skip rules as :func:`flow_direction_vectors` up to direction sign.
    If a later arrow-geometry step drops an edge, the caller must compare
    lengths before attaching hover columns.
    """
    from haemolynx.visualization.flow_direction import edge_flow_direction_sign

    kept: list[int] = []
    drawable = 0
    for _u, _v, _key, data in _iter_edges(graph):
        try:
            edge_polyline(graph, _u, _v, data)
        except ValueError:
            continue
        if edge_flow_direction_sign(data) is not None:
            kept.append(drawable)
        drawable += 1
    return np.asarray(kept, dtype=int)


def _limits(values: np.ndarray) -> tuple[float, float] | None:
    """nan-aware colour limits, or None when there is nothing finite to scale."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    low, high = float(finite.min()), float(finite.max())
    return (low, high) if high > low else (low, low + 1.0)


#: Signed normalised axis components of flow direction; always span [-1, 1].
FLOW_DIR_COLUMNS = frozenset({"flow_dir_z", "flow_dir_y", "flow_dir_x"})

#: Cyclic azimuth heading in the y-x plane; always spans [0, 360).
FLOW_HEADING_COLUMN = "flow_heading_deg"

#: Sentinel feature for the 3D RGB direction map (R=x, G=y, B=z).
FLOW_DIR_RGB_COLUMN = "flow_dir_rgb"


def _flow_dir_contrast_limits(values: np.ndarray) -> tuple[float, float]:
    """Colour limits for a ``flow_dir_*`` column.

    Components are unit vectors, so the meaningful range is [-1, 1]. Using the
    data min/max alone collapses axis-aligned flows to one colour when every
    arrow shares the same component (e.g. all ``flow_dir_z == 1``).
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0)
    low = min(-1.0, float(finite.min()))
    high = max(1.0, float(finite.max()))
    if high <= low:
        return (-1.0, 1.0)
    return (low, high)


def _flow_heading_contrast_limits(values: np.ndarray) -> tuple[float, float]:
    """Fixed cyclic range for ``flow_heading_deg``; do not autoscale."""
    _ = values
    return (0.0, 360.0)


def copy_graph(graph: Any) -> Any | None:
    """A pickle round-trip copy, or None when the graph will not pickle.

    The one place this happens -- checkpoints (stage_checkpoints.py) and run
    snapshots (run_snapshot.py) both import this rather than keeping their
    own copy, so a future change to how a graph is safely duplicated only
    has to land once.
    """
    if graph is None:
        return None
    try:
        return pickle.loads(pickle.dumps(graph))
    except Exception:  # noqa: BLE001 - a loaded run can still show stored layers
        logger.exception("could not pickle graph copy")
        return None


# --- one stage at a time -----------------------------------------------------


class ResultLayers:
    """Turns each finished stage's output into the layers it should show.

    Stateful, because a stage's output does not always carry what a layer
    needs: `BoundaryNodes` is four lists of node ids and `Solution.pressure` is
    ordered by `node_list`, while the positions for both live on the graph an
    earlier stage produced. So the graph is remembered as it goes past.
    """

    def __init__(
        self,
        *,
        prefix: str = PREFIX,
        show_steps: bool = False,
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        self.prefix = prefix
        #: Redraw the vessels after each of graph building's eleven topology
        #: steps. Off by default: it is eleven extra rebuilds of the geometry
        #: in the middle of the slowest stage.
        self.show_steps = show_steps
        #: Run settings (Export-tab toggles such as ``flow_arrow_scale``).
        self.settings: dict[str, Any] = dict(settings or {})
        self._graph: Any | None = None
        #: Live network after ``assign_boundaries`` (post large-vessel cut when
        #: that ran). Later stages must alias this object; a pre-cut copy with
        #: the same or larger edge count must not replace it in napari.
        self._canonical_graph: Any | None = None
        self._voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._image_shape_z: int | None = None
        self._geometry_shown = False
        self._emitted: list[str] = []
        #: The fat catchment `skeletonise` produced, if thickness-gated
        #: skeletonisation was on -- read by `_skeleton_layer_options` on
        #: every later re-emission of the skeleton layer. Restored by
        #: `load_state` so a resumed/reloaded run does not silently disable
        #: the thick/thin debug toggle for a run that genuinely used it.
        self._thick_vessel_mask: np.ndarray | None = None
        #: The pre-cleanup segmented mask, if a segmentation_cleanup_* step
        #: ran -- read by `_segmentation_cleanup_image_options` on every
        #: later re-emission of the IMAGE layer. Same restore-on-load
        #: treatment as `_thick_vessel_mask`.
        self._raw_segmented_image: np.ndarray | None = None

    @property
    def emitted(self) -> tuple[str, ...]:
        """Every layer name produced so far, in order."""
        return tuple(self._emitted)

    def reset(self) -> None:
        """Forget the run so far, so the next one starts from nothing.

        The graph is remembered as it goes past, which is the whole reason this
        class has state -- and a run that was stopped part-way leaves one
        behind. Without this, the stage after a restart would be drawn against
        the abandoned run's graph: `BoundaryNodes` is node ids, and they would
        be looked up in the wrong network.
        """
        self._graph = None
        self._canonical_graph = None
        self._voxel_size_zyx = (1.0, 1.0, 1.0)
        self._image_shape_z = None
        self._geometry_shown = False
        self._emitted = []
        self._thick_vessel_mask = None
        self._raw_segmented_image = None

    def export_state(self) -> dict[str, Any]:
        """Pickle-safe copy of the memory a loaded run needs to look finished."""
        graph = copy_graph(self._graph)
        if self._canonical_graph is self._graph:
            canonical = graph
        else:
            canonical = copy_graph(self._canonical_graph)
        thick_vessel_mask = self._thick_vessel_mask
        raw_segmented_image = self._raw_segmented_image
        return {
            "graph": graph,
            "canonical_graph": canonical,
            "voxel_size_zyx": tuple(float(v) for v in self._voxel_size_zyx),
            "image_shape_z": self._image_shape_z,
            "geometry_shown": bool(self._geometry_shown),
            "emitted": tuple(self._emitted),
            "settings": dict(self.settings),
            "show_steps": bool(self.show_steps),
            "thick_vessel_mask": (
                None if thick_vessel_mask is None else np.array(thick_vessel_mask, copy=True)
            ),
            "raw_segmented_image": (
                None if raw_segmented_image is None
                else np.array(raw_segmented_image, copy=True)
            ),
        }

    def load_state(self, state: Mapping[str, Any] | None) -> None:
        """Restore :meth:`export_state` so later stages and colouring match."""
        if not state:
            self.reset()
            return
        graph = copy_graph(state.get("graph"))
        canonical = state.get("canonical_graph")
        if canonical is state.get("graph"):
            canonical = graph
        else:
            canonical = copy_graph(canonical)
        self._graph = graph
        self._canonical_graph = canonical
        voxel = state.get("voxel_size_zyx") or (1.0, 1.0, 1.0)
        self._voxel_size_zyx = tuple(float(v) for v in voxel)
        self._image_shape_z = state.get("image_shape_z")
        self._geometry_shown = bool(state.get("geometry_shown"))
        self._emitted = list(state.get("emitted") or ())
        if state.get("settings") is not None:
            self.settings = dict(state["settings"])
        if "show_steps" in state:
            self.show_steps = bool(state["show_steps"])
        thick_vessel_mask = state.get("thick_vessel_mask")
        self._thick_vessel_mask = (
            None if thick_vessel_mask is None else np.array(thick_vessel_mask)
        )
        raw_segmented_image = state.get("raw_segmented_image")
        self._raw_segmented_image = (
            None if raw_segmented_image is None else np.array(raw_segmented_image)
        )

    def image_z_extent_um(self) -> float | None:
        """Physical Z span of the image stack once ``skeletonise`` has run."""
        if self._image_shape_z is None:
            return None
        return image_z_extent_um(self._voxel_size_zyx, self._image_shape_z)

    def stage_finished(self, stage: str, output: Any) -> StageLayers:
        """The layers for *stage*, built now, from *output* as it is now."""
        if stage.startswith(TOPOLOGY_STEP):
            return self._from_topology_step(stage[len(TOPOLOGY_STEP):], output)
        builder = _BUILDERS.get(stage)
        title = _title_for(stage)
        if builder is None:
            return StageLayers(stage=stage, title=title, note="")
        group = builder(self, output)
        for spec in group.layers:
            if spec.name not in self._emitted:
                self._emitted.append(spec.name)
        return group

    # -- the columns a graph can offer right now, for the colour-by control --

    def colour_options(self) -> list[str]:
        """Edge columns the remembered graph actually carries a value for."""
        if self._graph is None:
            return []
        return available_edge_columns(self._graph, self.settings)

    # -- builders ---------------------------------------------------------

    def _sync_graph_from_output(self, output: Any) -> Any | None:
        """Align ``_graph`` with a finished stage's live network when it carries one.

        After ``assign_boundaries`` the canonical graph is the post-cut network.
        A stale pre-cut copy handed back by a later stage must not regrow interior
        vessels in napari — even when it happens to carry the same edge count.
        """
        graph = getattr(output, "graph", None)
        if graph is None:
            return self._graph
        canonical = self._canonical_graph
        if canonical is not None and graph is not canonical:
            if graph.number_of_edges() >= canonical.number_of_edges():
                self._graph = canonical
                return self._graph
        self._graph = graph
        return self._graph

    def _vessel_layers(self, stage: str) -> tuple[LayerSpec, ...]:
        """The vessels Vectors (with polyline hover) and hidden label points.

        Always emits vessels / vessel-label layers (possibly empty). After a
        large-vessel volume cut the graph can lose every edge; omitting the
        specs would leave the pre-cut Vectors layer on screen and look like the
        cut kept interior geometry.
        """
        graph = self._graph
        assert graph is not None
        paths, identity = edge_polylines(graph)

        columns = {name: identity[name] for name in ("edge_index", "u", "v", "key")}
        # Every column every time, including the ones no stage has filled yet:
        # those come back NaN. napari's own "edge feature:" dropdown reads
        # `features.columns` once, when the layer controls are built, and never
        # listens for a features change (`qt_edge_color.py` calls `addItems` in
        # `__init__` and connects only to the edge_color events). So a column
        # that appears later is invisible there forever -- which is exactly what
        # happened to flow and pressure, written by the last stage, long after
        # the vessels layer was made.
        columns.update(edge_features(graph, edge_columns_for_settings(self.settings)))
        _enrich_flow_colour_columns(
            columns, graph,
            include_axis_components=bool(self.settings.get("flow_direction_colouring", True)),
        )

        vectors, owner = polylines_to_vectors(paths)
        per_segment = {
            name: np.asarray(values)[owner] for name, values in columns.items()
        }

        colour_by = DEFAULT_VESSEL_COLOUR.get(stage)
        if colour_by is not None and colour_by not in columns:
            colour_by = None
        elif (
            colour_by is not None
            and colour_by not in TEXT_COLUMNS
            and colour_by in columns
        ):
            numeric = np.asarray(columns[colour_by], dtype=float)
            if numeric.size == 0 or not np.any(np.isfinite(numeric)):
                colour_by = "branch_order" if "branch_order" in columns else None

        hover_columns, hover_options = _branch_hover_columns(graph, owner)
        if hover_columns:
            per_segment.update(hover_columns)

        midpoints = midpoints_of(paths)
        layers: list[LayerSpec] = [
            LayerSpec(
                kind="vectors",
                name=VESSELS,
                data=vectors,
                features=per_segment,
                colour_by=colour_by,
                **_colouring(per_segment, colour_by),
                options={
                    "vector_style": "line",
                    "edge_width": 0.6,
                    "out_of_slice_display": True,
                    **hover_options,
                },
            ),
            # Identity table for colour-by columns that are per-edge rather
            # than per-segment. Hidden; hover hit-testing is on the Vectors.
            LayerSpec(
                kind="points",
                name=VESSEL_LABELS,
                data=midpoints,
                features=columns,
                visible=False,
                options={
                    "size": VESSEL_LABEL_POINT_SIZE,
                    "out_of_slice_display": True,
                },
            ),
        ]
        return tuple(layers)

    def layers_for_graph(
        self,
        graph: Any,
        *,
        stage: str = "edit",
        draft_points_um: Sequence[Sequence[float]] | None = None,
    ) -> StageLayers:
        """Rebuild the vessels/nodes layers straight from *graph*.

        Every other builder here reads a real stage's ``output`` object; the
        "Edit" window has no such object -- just a graph it just mutated --
        so this is the one entry point that takes a graph directly instead.
        The NODES layer built here mirrors :meth:`_from_build_network`'s own
        (``node_id``/``degree`` features, coloured by degree), since an edit
        is closest in spirit to a freshly built network, not a haemodynamics
        result the node layer would otherwise carry.

        *draft_points_um* is the "Add branch" draft's own path so far, if
        any -- shown as a distinct :data:`EDIT_DRAFT` layer, since the draft
        is not part of *graph* until it commits. The caller is responsible
        for removing that layer once there is no longer a draft to show (see
        ``gui._widget``'s own handling); this only ever adds or updates it.
        """
        self._graph = graph
        layers: list[LayerSpec] = list(self._vessel_layers(stage))
        if draft_points_um is not None and len(draft_points_um) >= 2:
            layers.append(
                LayerSpec(
                    kind="shapes",
                    name=EDIT_DRAFT,
                    data=[np.asarray(draft_points_um, dtype=float)],
                    options={
                        "shape_type": ["path"],
                        "edge_color": "yellow",
                        "edge_width": 2.0,
                        "opacity": 0.9,
                    },
                )
            )
        points, ids = node_points(graph)
        degrees = np.asarray([graph.degree(node_id) for node_id in ids], dtype=float)
        layers.append(
            LayerSpec(
                kind="points",
                name=NODES,
                data=points,
                features={
                    "node_id": ids,
                    "degree": degrees,
                    "pressure": np.full(len(ids), np.nan),
                },
                colour_by="degree",
                colour_kind="continuous",
                contrast_limits=_limits(degrees),
                options={"size": NODE_POINT_SIZE, "out_of_slice_display": True},
            )
        )
        return StageLayers(
            stage=stage,
            title="Edit",
            layers=tuple(layers),
            note=f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} vessels (edited).",
        )

    def _from_topology_step(self, label: str, graph: Any) -> StageLayers:
        """The graph part-way through its repair, when asked for.

        The output here is the graph itself, mid-repair -- `build_network` has
        not returned, so there is no VesselNetwork yet -- and it will change
        again on the next step. Nothing is remembered from it.
        """
        if not self.show_steps or graph is None:
            return StageLayers(stage=f"{TOPOLOGY_STEP}{label}", title=label)
        held = self._graph
        self._graph = graph
        try:
            layers = self._vessel_layers("build_network")
        finally:
            self._graph = held
        return StageLayers(
            stage=f"{TOPOLOGY_STEP}{label}",
            title=label,
            layers=layers,
            note=f"{label}: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} vessels.",
        )

    def _skeleton_layer_options(self) -> dict[str, Any]:
        """Extra options every re-emission of the skeleton layer carries.

        Read by _store_thick_thin_skeleton_metadata in _widget.py, which
        pops "thick_vessel_mask" before napari ever sees it as a kwarg (see
        _THICK_THIN_OPTION_KEYS there) and stashes it on the layer so its
        thick/thin debug toggle survives every later stage re-emitting this
        same layer, not only the skeletonise stage that first computed it.
        """
        thick_vessel_mask = getattr(self, "_thick_vessel_mask", None)
        if thick_vessel_mask is None:
            return {}
        return {"thick_vessel_mask": thick_vessel_mask}

    def _segmentation_cleanup_image_options(self) -> dict[str, Any]:
        """Extra options every re-emission of the IMAGE layer carries, read
        by _store_segmentation_cleanup_metadata in _widget.py -- same shape
        as _skeleton_layer_options, for the raw-vs-corrected mask toggle."""
        raw_segmented_image = getattr(self, "_raw_segmented_image", None)
        if raw_segmented_image is None:
            return {}
        return {"raw_segmented_image": raw_segmented_image}

    def _from_segment(self, output: Any) -> StageLayers:
        """Nothing to draw: the stage settles which file to read, not its content."""
        path = getattr(output, "image_path", None)
        return StageLayers(
            stage="segment",
            title=_title_for("segment"),
            note=f"Reading {path}. Its content arrives with the next stage."
            if path
            else "",
        )

    def _from_skeletonise(self, output: Any) -> StageLayers:
        scale = tuple(float(v) for v in getattr(output, "voxel_size_zyx", (1.0, 1.0, 1.0)))
        self._voxel_size_zyx = scale  # type: ignore[assignment]
        image = getattr(output, "image", None)
        skeleton = getattr(output, "skeleton", None)
        self._image_shape_z = int(np.asarray(image).shape[0]) if image is not None else None
        layers: list[LayerSpec] = []
        if image is not None:
            display_image = image
            value_range = binary_value_range(image)
            if value_range is not None:
                # binary_value_range only tells us the image is two-valued,
                # not which value is foreground -- an ilastik export's class
                # labels carry no relation to which one is the minority
                # (e.g. 1=vessel/10%, 2=background/90% is exactly as likely
                # as the reverse). Using (low, high) directly as
                # contrast_limits assumes low=background, which silently
                # inverts the mask whenever the smaller label number happens
                # to be the foreground: the real vessel becomes transparent
                # and the majority background becomes the opaque colour,
                # rendering as a solid box. Binarizing through the same
                # canonical foreground split the pipeline itself uses keeps
                # the display always agreeing with what gets skeletonized.
                from haemolynx.io.load import _to_binary_volume_for_skeletonization

                display_image = _display_owned(
                    _to_binary_volume_for_skeletonization(image, use_memmap=_on_disk(image))
                )
                value_range = (0.0, 1.0)
            else:
                binarized = binarize_for_display(image)
                if binarized is not None:
                    display_image = binarized
                    value_range = (0.0, 1.0)
            if value_range is not None:
                image_options: dict[str, Any] = {
                    **BINARY_IMAGE_VOLUME_OPTIONS,
                    "mask_colour": SEGMENTED_IMAGE_COLOUR,
                }
                # raw_segmented_image is always a canonical boolean mask, so
                # the corrected-vs-raw comparison only makes sense when the
                # layer is actually rendered in this translucent binary-mask
                # style -- the grayscale fallback below is a genuine
                # continuous image, where a boolean diff means nothing.
                self._raw_segmented_image = getattr(output, "raw_segmented_image", None)
                image_options.update(self._segmentation_cleanup_image_options())
            else:
                image_options = {"blending": "additive", "colormap": "gray"}
            layers.append(
                LayerSpec(kind="image", name=IMAGE, data=display_image, scale=scale,
                          options=image_options, contrast_limits=value_range)
            )
        if skeleton is not None:
            self._thick_vessel_mask = getattr(output, "thick_vessel_mask", None)
            layers.append(
                LayerSpec(
                    kind="labels", name=SKELETON, data=skeleton, scale=scale,
                    options=self._skeleton_layer_options(),
                )
            )
            self._skeleton = skeleton
        return StageLayers(
            stage="skeletonise",
            title=_title_for("skeletonise"),
            layers=tuple(layers),
            note=f"Voxel size (z, y, x): {scale}",
        )

    def _from_build_network(self, output: Any) -> StageLayers:
        graph = getattr(output, "graph", None)
        self._graph = graph
        layers: list[LayerSpec] = []

        if graph is not None:
            layers.extend(self._vessel_layers("build_network"))
            skeleton = getattr(getattr(output, "volume", None), "skeleton", None)
            if skeleton is not None and graph.number_of_edges() > 0:
                scale = tuple(float(v) for v in self._voxel_size_zyx)
                layers.append(
                    LayerSpec(
                        kind="labels",
                        name=SKELETON,
                        data=skeleton,
                        scale=scale,
                        options=self._skeleton_layer_options(),
                        visible=False,
                    )
                )
            points, ids = node_points(graph)
            degrees = np.asarray([graph.degree(node_id) for node_id in ids], dtype=float)
            layers.append(
                LayerSpec(
                    kind="points", name=NODES, data=points,
                    # `pressure` is declared empty here for the same reason the
                    # vessel columns are: napari's layer controls read the
                    # column list once, so a column the solve adds later would
                    # never reach the dropdown on the left.
                    features={"node_id": ids, "degree": degrees,
                              "pressure": np.full(len(ids), np.nan)},
                    colour_by="degree", colour_kind="continuous",
                    contrast_limits=_limits(degrees),
                    options={
                        "size": NODE_POINT_SIZE,
                        "out_of_slice_display": True,
                    },
                    # Start hidden -- a busy point cloud on first view of a new
                    # run. Only the first emission sets this: later stages'
                    # own NODES specs default to visible=True, which restores
                    # whatever the user last set rather than re-hiding it (see
                    # _add_or_update's saved_visible handling in _widget.py).
                    visible=False,
                )
            )

        layers.extend(
            vessel_mask_volume_layers(output, voxel_size_zyx=self._voxel_size_zyx)
        )

        edges = graph.number_of_edges() if graph is not None else 0
        nodes = graph.number_of_nodes() if graph is not None else 0
        first_geometry = not self._geometry_shown and bool(layers)
        self._geometry_shown = self._geometry_shown or first_geometry
        return StageLayers(
            stage="build_network",
            title=_title_for("build_network"),
            layers=tuple(layers),
            note=f"{nodes} nodes, {edges} vessels.",
            # A network drawn in a 2D slice is a handful of dots; show it whole.
            ndisplay=3 if first_geometry else None,
        )

    def _from_assign_boundaries(self, output: Any) -> StageLayers:
        # Prefer the live graph returned by assign_boundaries (post cut when
        # enabled). Falling back to the remembered build_network graph keeps
        # older SimpleNamespace test fixtures working.
        graph = getattr(output, "graph", None)
        if graph is not None:
            self._graph = graph
            self._canonical_graph = graph

        roles = {
            "inlet": getattr(output, "inlet_nodes", ()) or (),
            "outlet": getattr(output, "outlet_nodes", ()) or (),
            "arteriole_boundary": getattr(output, "arteriole_boundary_nodes", ()) or (),
            "venule_boundary": getattr(output, "venule_boundary_nodes", ()) or (),
        }
        layers: list[LayerSpec] = []
        if self._graph is not None:
            # Refresh vessels and nodes so a large-vessel volume cut does not
            # leave pre-cut interior geometry on screen — including when the
            # post-cut graph is empty (emit empty layers to clear the viewer).
            layers.extend(self._vessel_layers("assign_boundaries"))
            skeleton = getattr(self, "_skeleton", None)
            if skeleton is not None:
                layers.append(
                    LayerSpec(
                        kind="labels",
                        name=SKELETON,
                        data=skeleton,
                        scale=tuple(float(v) for v in self._voxel_size_zyx),
                        options=self._skeleton_layer_options(),
                        visible=False,
                    )
                )
            points, ids = node_points(self._graph)
            if len(points):
                degrees = np.asarray(
                    [self._graph.degree(node_id) for node_id in ids], dtype=float
                )
                layers.append(
                    LayerSpec(
                        kind="points",
                        name=NODES,
                        data=points,
                        features={
                            "node_id": ids,
                            "degree": degrees,
                            "pressure": np.full(len(ids), np.nan),
                        },
                        colour_by="degree",
                        colour_kind="continuous",
                        contrast_limits=_limits(degrees),
                        options={
                            "size": NODE_POINT_SIZE,
                            "out_of_slice_display": True,
                        },
                    )
                )
            else:
                layers.append(
                    LayerSpec(
                        kind="points",
                        name=NODES,
                        data=np.empty((0, 3), dtype=float),
                        features={
                            "node_id": np.asarray([], dtype=object),
                            "degree": np.asarray([], dtype=float),
                            "pressure": np.asarray([], dtype=float),
                        },
                        options={
                            "size": NODE_POINT_SIZE,
                            "out_of_slice_display": True,
                        },
                    )
                )
            positions: list[np.ndarray] = []
            labels: list[str] = []
            boundary_ids: list[Any] = []
            for role, node_ids in roles.items():
                role_points, kept = node_points(self._graph, list(node_ids))
                if len(role_points):
                    positions.append(role_points)
                    labels.extend([role] * len(kept))
                    boundary_ids.extend(kept.tolist())
            if positions:
                role_column = np.asarray(labels, dtype=object)
                layers.append(
                    LayerSpec(
                        kind="points", name=BOUNDARY_NODES,
                        data=np.concatenate(positions, axis=0),
                        features={"role": role_column,
                                  "node_id": np.asarray(boundary_ids, dtype=object)},
                        colour_by="role", colour_kind="categorical",
                        colour_cycle=role_colours(),
                        options={
                            "size": BOUNDARY_NODE_POINT_SIZE,
                            "out_of_slice_display": True,
                        },
                        # Start hidden -- this is the only stage that ever
                        # builds this layer, so there is no later spec whose
                        # own visible=True would need to preserve a user
                        # toggle (see the matching NODES comment above).
                        visible=False,
                    )
                )

        if getattr(output, "large_arteriole_mask", None) is not None or getattr(
            output, "large_venule_mask", None
        ) is not None:
            layers.extend(
                vessel_mask_volume_layers(output, voxel_size_zyx=self._voxel_size_zyx)
            )

        counts = ", ".join(f"{len(ids)} {role}" for role, ids in roles.items() if ids)
        return StageLayers(
            stage="assign_boundaries",
            title=_title_for("assign_boundaries"),
            layers=tuple(layers),
            note=counts or "No boundary nodes found.",
        )

    def _from_assign_diameters(self, output: Any) -> StageLayers:
        self._sync_graph_from_output(output)
        layers = list(self._vessel_layers("assign_diameters"))
        points, features = pericyte_points(self._graph) if self._graph is not None else (
            np.empty((0, 3)), {}
        )
        if len(points):
            layers.append(
                LayerSpec(
                    kind="points", name=PERICYTES, data=points, features=features,
                    colour_by="branch_order", colour_kind="categorical",
                    colour_cycle=colour_cycle_for(features.get("branch_order", ())),
                    options={
                        "size": PERICYTE_POINT_SIZE,
                        "out_of_slice_display": True,
                    },
                )
            )
        fwhm_raw = getattr(output, "fwhm_raw", None)
        if fwhm_raw is not None:
            layers.append(
                LayerSpec(
                    kind="image",
                    name=FWHM_RAW,
                    data=fwhm_raw,
                    scale=tuple(float(v) for v in self._voxel_size_zyx),
                    options={"blending": "additive", "colormap": "gray", "opacity": 0.8},
                )
            )
        if self._graph is not None:
            for name, paths, colour, width in (
                (FWHM_PROFILES, fwhm_profile_polylines(self._graph), "cyan", 0.35),
                (FWHM_WIDTHS, fwhm_measured_polylines(self._graph), "magenta", 0.7),
            ):
                if not paths:
                    continue
                vectors, _owner = polylines_to_vectors(paths)
                layers.append(
                    LayerSpec(
                        kind="vectors",
                        name=name,
                        data=vectors,
                        options={
                            "vector_style": "line",
                            "edge_width": width,
                            "edge_color": colour,
                            "out_of_slice_display": True,
                        },
                    )
                )
        orders = sorted(
            {str(o) for o in edge_features(self._graph, ["branch_order"])["branch_order"] if o}
        ) if self._graph is not None else []
        sources = {}
        if self._graph is not None:
            for _u, _v, _key, data in _iter_edges(self._graph):
                source = data.get("diameter_source")
                if source:
                    sources[str(source)] = sources.get(str(source), 0) + 1
        source_note = ", ".join(f"{count} {name}" for name, count in sorted(sources.items()))
        note = f"{len(orders)} branch orders"
        if source_note:
            note += f", {source_note}"
        if len(points):
            note += f", {len(points)} pericytes"
        return StageLayers(
            stage="assign_diameters",
            title=_title_for("assign_diameters"),
            layers=tuple(layers),
            note=note,
        )

    def _from_build_haemodynamic_model(self, output: Any) -> StageLayers:
        """No new layer: this stage returns the object it was given.

        What it does establish is that every edge now carries a resistance, so
        the honest thing to show is the network repainted by it -- the one stage
        whose visible effect is a change of view rather than new data.
        """
        graph = self._sync_graph_from_output(output)
        with_resistance = 0
        total = 0
        if graph is not None:
            for _u, _v, _key, data in _iter_edges(graph):
                total += 1
                if data.get("resistance") is not None:
                    with_resistance += 1
        return StageLayers(
            stage="build_haemodynamic_model",
            title=_title_for("build_haemodynamic_model"),
            recolour=((VESSELS, "resistance"),),
            note=f"Resistance on {with_resistance} of {total} vessels.",
        )

    def _from_solve(self, output: Any) -> StageLayers:
        self._sync_graph_from_output(output)
        layers = list(self._vessel_layers("solve"))
        skeleton = getattr(self, "_skeleton", None)
        if skeleton is not None:
            layers.append(
                LayerSpec(
                    kind="labels",
                    name=SKELETON,
                    data=skeleton,
                    scale=tuple(float(v) for v in self._voxel_size_zyx),
                    options=self._skeleton_layer_options(),
                    visible=False,
                )
            )
        pressure = getattr(output, "pressure", None)
        node_list = list(getattr(output, "node_list", ()) or ())
        if self._graph is not None and pressure is not None and node_list:
            # `pressure` is ordered by `node_list`, not by node id: pair them
            # before dropping any node that has no position.
            by_node = {node_id: float(p) for node_id, p in zip(node_list, pressure)}
            points, ids = node_points(self._graph, node_list)
            values = np.asarray([by_node[node_id] for node_id in ids], dtype=float)
            if len(points):
                layers.append(
                    LayerSpec(
                        kind="points", name=NODES, data=points,
                        features={"node_id": ids, "pressure": values,
                                  "degree": np.asarray(
                                      [self._graph.degree(n) for n in ids], dtype=float)},
                        colour_by="pressure", colour_kind="continuous",
                        contrast_limits=_limits(values),
                        options={
                            "size": NODE_POINT_SIZE,
                            "out_of_slice_display": True,
                        },
                    )
                )
        equivalent = getattr(output, "equivalent_resistance", None)
        note = "Solved."
        if equivalent is not None:
            note = f"Equivalent resistance {float(equivalent):.4e} Pa.s/m^3."
        return StageLayers(
            stage="solve", title=_title_for("solve"), layers=tuple(layers), note=note
        )

    def _perturbation_layers(self, result: Any) -> tuple[LayerSpec, ...]:
        """One perturbation's own vessels and nodes, named after it.

        Its own copy of the network, so the layers are built from
        `result.graph` and the baseline this class is holding is left alone --
        the colour-by dropdown goes on offering the baseline's columns, and the
        baseline's own layers are not touched.

        Hidden, and deliberately. A perturbation is the same geometry as the
        baseline with different numbers on it, so a visible one lies exactly on
        top of the vessels it is meant to be compared with and whichever was
        added last wins. Ticking one on is the comparison.
        """
        graph = result.graph
        vessels_name, nodes_name = perturbation_layer_names(result.name)
        paths, identity = edge_polylines(graph)
        if not paths:
            return ()

        columns = {name: identity[name] for name in ("edge_index", "u", "v", "key")}
        columns.update(edge_features(graph, edge_columns_for_settings(self.settings)))
        # The same derived flow columns (log10, direction) and hover tooltips
        # the baseline's vessels get, from this perturbation's own flows.
        _enrich_flow_colour_columns(
            columns, graph,
            include_axis_components=bool(self.settings.get("flow_direction_colouring", True)),
        )
        vectors, owner = polylines_to_vectors(paths)
        per_segment = {
            name: np.asarray(values)[owner] for name, values in columns.items()
        }
        hover_columns, hover_options = _branch_hover_columns(graph, owner)
        if hover_columns:
            per_segment.update(hover_columns)
        colour_by = PERTURBATION_COLOUR if PERTURBATION_COLOUR in columns else None

        layers = [
            LayerSpec(
                kind="vectors",
                name=vessels_name,
                data=vectors,
                features=per_segment,
                colour_by=colour_by,
                **_colouring(per_segment, colour_by),
                visible=False,
                options={"vector_style": "line", "edge_width": 0.6,
                         "out_of_slice_display": True, **hover_options},
            )
        ]

        points, ids = node_points(graph)
        if len(points):
            # `set_edge_flows` writes `pressure` onto every node it solved, so
            # a perturbation's pressures travel on its own graph -- there is no
            # `Solution` to read them out of here.
            pressures = np.asarray(
                [
                    float(graph.nodes[node_id].get("pressure", np.nan))
                    for node_id in ids
                ],
                dtype=float,
            )
            layers.append(
                LayerSpec(
                    kind="points",
                    name=nodes_name,
                    data=points,
                    features={"node_id": ids, "pressure": pressures,
                              "degree": np.asarray(
                                  [graph.degree(node_id) for node_id in ids],
                                  dtype=float)},
                    colour_by="pressure",
                    colour_kind="continuous",
                    contrast_limits=_limits(pressures),
                    visible=False,
                    options={
                        "size": NODE_POINT_SIZE,
                        "out_of_slice_display": True,
                    },
                )
            )
        layers.extend(
            self._flow_direction_layers(
                graph, name=perturbation_flow_direction_layer_name(result.name), visible=False
            )
        )
        return tuple(layers)

    def _sweep_perturbation_layers(self, result: Any) -> tuple[LayerSpec, ...]:
        """One Vectors layer for a sweep: geometry once, flows from the grid.

        Initial colouring is grid point 0; the widget attaches slider(s) that
        swap ``flow_abs`` (and signed / drop columns) via *segment_owner*
        without rebuilding polylines.
        """
        graph = result.graph
        sweep = getattr(result, "sweep_flows", None)
        if graph is None or sweep is None:
            return ()

        vessels_name, _nodes_name = perturbation_layer_names(result.name)
        paths, identity = edge_polylines(graph)
        if not paths:
            return ()

        edge_index = np.asarray(identity["edge_index"], dtype=int)
        columns = {name: identity[name] for name in ("edge_index", "u", "v", "key")}
        # Geometry / diameter columns from the retained graph; flow columns
        # come from the sweep grid so slider 0 matches the retained arrays.
        # The retained graph is the unperturbed baseline, so any column a
        # grid point changes is left out rather than shown with baseline
        # values: flow magnitudes (transit times) under any sweep, and
        # diameters, resistances and every solved-network analysis once a
        # sweep also moves geometry.
        stale = set(SWEEP_FLOW_MAGNITUDE_COLUMNS)
        if tuple(sweep.axis_names) != ("inlet_pressure_pa",):
            stale.update(SWEEP_GEOMETRY_DEPENDENT_COLUMNS)
            stale.update(
                name for name, stage in OPTIONAL_EDGE_COLUMNS.items()
                if stage == "export_results"
            )
        non_flow = [
            name
            for name in edge_columns_for_settings(self.settings)
            if name not in ("flow_abs", "flow_abs_log10", "flow_signed", "pressure_drop",
                            "pressure_u", "pressure_v")
            and name not in stale
        ]
        columns.update(edge_features(graph, non_flow))

        flow0 = np.asarray(sweep.flow_abs_at(*([0] * len(sweep.axis_names))), dtype=float)
        columns["flow_abs"] = flow0[edge_index]
        from haemolynx.haemodynamics.resistance import flow_abs_log10_value

        columns["flow_abs_log10"] = np.asarray(
            [flow_abs_log10_value(v) for v in flow0[edge_index]], dtype=float
        )
        signed0 = sweep.flow_signed_at(*([0] * len(sweep.axis_names)))
        directions = None
        if signed0 is not None:
            columns["flow_signed"] = np.asarray(signed0, dtype=float)[edge_index]
            # Direction columns follow the slider like the flows do: both
            # senses once here, the grid point's own signs picking per edge.
            from haemolynx.visualization.flow_direction import edge_flow_direction_columns

            directions = tuple(
                {
                    name: values
                    for name, values in edge_flow_direction_columns(
                        graph, sign_override=sign
                    ).items()
                    if self.settings.get("flow_direction_colouring", True)
                    or name not in FLOW_DIR_COLUMNS
                }
                for sign in (1, -1)
            )
            columns.update(sweep_direction_columns(directions, columns["flow_signed"]))
        drop0 = sweep.pressure_drop_at(*([0] * len(sweep.axis_names)))
        if drop0 is not None:
            columns["pressure_drop"] = np.asarray(drop0, dtype=float)[edge_index]

        vectors, owner = polylines_to_vectors(paths)
        per_segment = {
            name: np.asarray(values)[owner] for name, values in columns.items()
        }
        colour_by = PERTURBATION_COLOUR if PERTURBATION_COLOUR in columns else None
        colouring = _colouring(per_segment, colour_by)
        # Stable contrast across the whole grid so sliding does not re-stretch.
        global_limits = sweep.global_flow_abs_limits()
        if colour_by == "flow_abs" and global_limits is not None:
            colouring = {**colouring, "contrast_limits": global_limits}

        return (
            LayerSpec(
                kind="vectors",
                name=vessels_name,
                data=vectors,
                features=per_segment,
                colour_by=colour_by,
                **colouring,
                visible=False,
                options={"vector_style": "line", "edge_width": 0.6,
                         "out_of_slice_display": True},
                sweep=sweep,
                segment_owner=owner,
                sweep_edge_index=edge_index,
                sweep_directions=directions,
            ),
        )

    def _from_run_perturbations(self, output: Any) -> StageLayers:
        """One layer set per perturbation that produced a network or sweep grid.

        Non-sweeps get vessels+nodes from their graph. Sweeps get one Vectors
        layer backed by retained per-grid flow arrays (slider UI in the widget).
        """
        from haemolynx.haemodynamics.perturbations import is_sweep_perturbation

        results = list(getattr(output, "solved", ()) or ())
        layers: list[LayerSpec] = []
        for result in results:
            if is_sweep_perturbation(getattr(result, "type", "")):
                layers.extend(self._sweep_perturbation_layers(result))
            else:
                layers.extend(self._perturbation_layers(result))

        failures = list(getattr(output, "failures", ()) or ())
        layer_count = len([spec for spec in layers if spec.kind == "vectors"])
        note = (
            f"{layer_count} perturbation(s) re-solved; their flow layers are "
            "hidden, so tick one to compare it with the baseline."
            if layer_count
            else "No perturbation produced a network layer."
        )
        if failures:
            note += " Failed: " + ", ".join(
                f"{result.name} ({result.error})" for result in failures
            )
        return StageLayers(
            stage="run_perturbations",
            title=_title_for("run_perturbations"),
            layers=tuple(layers),
            note=note,
        )

    def _flow_direction_layers(
        self, graph: Any = None, *, name: str = FLOW_DIRECTION, visible: bool = True
    ) -> tuple[LayerSpec, ...]:
        """Mid-edge flow arrows coloured by 3D direction RGB, or empty when none exist.

        The baseline's by default; a perturbation passes its own *graph* and
        layer *name*, hidden like the rest of its layers.
        """
        from haemolynx.visualization.flow_direction import flow_direction_vectors

        graph = self._graph if graph is None else graph
        if graph is None:
            return ()
        vectors, features = flow_direction_vectors(graph)
        if len(vectors) == 0:
            return ()
        if not self.settings.get("flow_direction_colouring", True):
            # The layer itself is unconditional (see flow_direction_colouring's
            # own help text) -- only the raw axis components are excluded from
            # the colour-by dropdown; flow_dir_rgb/flow_heading_deg (derived,
            # composite encodings, not "axis components") stay available.
            features = {k: v for k, v in features.items() if k not in FLOW_DIR_COLUMNS}
        colour_by = (
            FLOW_DIR_RGB_COLUMN
            if FLOW_DIR_RGB_COLUMN in features
            else (
                FLOW_HEADING_COLUMN
                if FLOW_HEADING_COLUMN in features
                else ("flow_abs" if "flow_abs" in features else None)
            )
        )
        scale = float(self.settings.get("flow_arrow_scale", 1.0))
        hover_index = _flow_direction_drawable_edge_index(graph)
        hover_columns, hover_options = (
            _branch_hover_columns(graph, hover_index)
            if len(hover_index) == len(vectors)
            else ({}, {})
        )
        if hover_columns:
            features = {**features, **hover_columns}
        return (
            LayerSpec(
                kind="vectors",
                name=name,
                data=vectors,
                features=features,
                colour_by=colour_by,
                **_colouring(features, colour_by),
                visible=visible,
                options={
                    "vector_style": "triangle",
                    "edge_width": 1.2,
                    "length": scale,
                    "out_of_slice_display": True,
                    **hover_options,
                },
            ),
        )

    def _from_export_results(self, output: Any) -> StageLayers:
        self._sync_graph_from_output(output)
        note = "Wrote the VTK, statistics and plots."
        # Re-emit the vessels layer so any attribute this stage itself wrote
        # (e.g. vascular_community, from graph.assign_vascular_communities)
        # actually reaches layer.features -- DEFAULT_VESSEL_COLOUR has no
        # "export_results" entry, so colour_by resolves to None here, which
        # _colour_layer treats as "leave the current colouring alone" (see
        # its own docstring): this refreshes features without disturbing
        # whatever solve, or the user, already coloured the layer by.
        flow_layers = self._flow_direction_layers()
        layers = self._vessel_layers("export_results") + flow_layers
        if flow_layers:
            note += f" Flow direction: {len(flow_layers[0].data)} arrows."
        else:
            note += " Flow direction layer skipped (no signed flows)."
        return StageLayers(
            stage="export_results",
            title=_title_for("export_results"),
            layers=layers,
            note=note,
        )


def _colouring(columns: Mapping[str, np.ndarray], colour_by: str | None) -> dict[str, Any]:
    """How to colour by *colour_by*: a cycle for text, limits for numbers."""
    if colour_by is None or colour_by not in columns:
        return {"colour_kind": "none"}
    values = columns[colour_by]
    if colour_by in TEXT_COLUMNS:
        return {"colour_kind": "categorical", "colour_cycle": colour_cycle_for(values)}
    if colour_by == FLOW_DIR_RGB_COLUMN:
        # Direct per-vector RGB; no 1D LUT or clim.
        return {"colour_kind": "direct"}
    if colour_by in FLOW_DIR_COLUMNS:
        return {
            "colour_kind": "continuous",
            "contrast_limits": _flow_dir_contrast_limits(values),
        }
    if colour_by == FLOW_HEADING_COLUMN:
        return {
            "colour_kind": "continuous",
            "contrast_limits": _flow_heading_contrast_limits(values),
        }
    return {"colour_kind": "continuous", "contrast_limits": _limits(values)}


def role_colours() -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
    """Inlets, outlets and vessel-type boundaries, told apart at a glance.

    Green in, red out: where flow enters and where it leaves is the thing
    being looked for, so those two carry the plainest colours and the two
    vessel-type boundaries take colours that cannot be mistaken for either.
    """
    return (
        ("inlet", (0.17, 0.63, 0.17, 1.0)),
        ("outlet", (0.84, 0.15, 0.16, 1.0)),
        ("arteriole_boundary", (1.0, 0.5, 0.05, 1.0)),
        ("venule_boundary", (0.58, 0.40, 0.74, 1.0)),
        ("large_vessel_inlet", (0.55, 0.0, 0.0, 1.0)),
        ("large_vessel_outlet", (0.03, 0.19, 0.42, 1.0)),
    )


def _title_for(stage: str) -> str:
    from haemolynx.pipeline.progress import STAGES

    for entry in STAGES:
        if entry.call == stage:
            return entry.title
    return stage


_BUILDERS = {
    "segment": ResultLayers._from_segment,
    "skeletonise": ResultLayers._from_skeletonise,
    "build_network": ResultLayers._from_build_network,
    "assign_boundaries": ResultLayers._from_assign_boundaries,
    "assign_diameters": ResultLayers._from_assign_diameters,
    "build_haemodynamic_model": ResultLayers._from_build_haemodynamic_model,
    "solve": ResultLayers._from_solve,
    "run_perturbations": ResultLayers._from_run_perturbations,
    "export_results": ResultLayers._from_export_results,
}
