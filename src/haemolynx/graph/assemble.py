"""Assemble a cleaned vascular graph from a 3D skeleton."""
from __future__ import annotations

import logging
from collections.abc import Callable
import networkx as nx
import numpy as np
from skan import csr

from haemolynx.preprocessing.bridge_mask_support import (
    DEFAULT_MAX_BACKGROUND_GAP_UM,
    DEFAULT_MIN_MASK_FRACTION,
    MaskSupport,
)

from ._helpers import points_inside_mask
from ._platform import skan_numba_warmup_skeleton
from .build import (
    MAX_GAP_BRIDGE_TURN_DEG,
    build_graph_segment_skan_stitched_loops,
    skan_skeleton,
)
from .collapse import collapse_node_clusters
from .direction_aware_collapse import (
    DEFAULT_MAX_RADIAL_DISPERSION,
    DEFAULT_MIN_DEGREE_FOR_DISPERSION_CHECK,
    collapse_node_clusters_direction_aware,
)
from .cartwheel_guard import DEFAULT_TANGENT_LENGTH_UM
from .persistence_collapse import (
    DEFAULT_SEARCH_RADIUS_MULTIPLE,
    collapse_node_clusters_persistence,
)
from .degree2 import smart_multigraph_degree2_removal
from .diagnostics import diagnose_degree2_nodes, format_degree2_diagnostics_report
from .mask_recovery import (
    DEFAULT_MIN_LENGTH_UM,
    DEFAULT_MIN_REGION_VOLUME_UM3,
    recover_uncovered_mask_vessels as _recover_uncovered_mask_vessels,
)
from .lumen_loops import remove_loops_inside_one_lumen
from .optimise import optimise_graph_topology_fixed, reconnect_orphan_and_dangling_nodes
from .prune import prune_vascular_stubs, remove_edges_for_self_connected_nodes
from .reconnect import reconnect_secondary_loop_edges

logger = logging.getLogger(__name__)

StepCallback = Callable[[nx.MultiGraph, str], None]

#: Every step `build_graph_from_skeleton` reports, in the order it runs them.
#: A progress bar has to know how many there will be before the first one
#: fires, and a snapshot writer has to know the labels are unique, so the list
#: is declared here rather than left implicit in the call order below.
#: `tests/test_graph_assemble.py` fails if a build stops matching it.
STEP_LABELS: tuple[str, ...] = (
    "build_graph_segment_skan_stitched_loops",
    "reconnect_secondary_loop_edges",
    "optimise_graph_topology_fixed",
    "smart_multigraph_degree2_removal_pass1",
    "collapse_node_clusters",
    "smart_multigraph_degree2_removal_post_collapse",
    "prune_vascular_stubs",
    "smart_multigraph_degree2_removal_post_prune",
    "remove_edges_for_self_connected_nodes",
    "reconnect_orphan_and_dangling_nodes",
    "recover_uncovered_mask_vessels",
    "remove_loops_inside_one_lumen",
    "prune_vascular_stubs_final",
    "smart_multigraph_degree2_removal_post_orphan_reconnect",
)


#: Where each label comes in the run, for the line every step logs. A step
#: names itself in that line, and `collapse_node_clusters` names itself in its
#: own summary too, so the `Step n/14` prefix is what tells the two apart.
_STEP_POSITIONS: dict[str, int] = {
    label: position for position, label in enumerate(STEP_LABELS, start=1)
}


def step_after(label: str) -> str | None:
    """The topology step that starts once *label* finishes: ``None`` after
    the last step, or for a label that is not one of :data:`STEP_LABELS`."""
    position = _STEP_POSITIONS.get(label)
    if position is None or position >= len(STEP_LABELS):
        return None
    return STEP_LABELS[position]


def _notify_step(
    G: nx.MultiGraph,
    label: str,
    step_callback: StepCallback | None,
) -> None:
    # Fourteen lines a run, ungated: what a step left behind is the answer to
    # "how many branches does the pipeline think there are", and asking for it
    # should not mean asking for the per-node detail as well. The line names
    # the step starting next: it is the last line until that step ends, and
    # read alone it blamed the step just finished for the next one's time.
    following = step_after(label)
    logger.info(
        "Step %d/%d %s: %d nodes / %d edges%s",
        _STEP_POSITIONS.get(label, 0),
        len(STEP_LABELS),
        label,
        G.number_of_nodes(),
        G.number_of_edges(),
        f"; running {following}" if following else "",
    )
    if step_callback is not None:
        step_callback(G, label)


def mask_radius_sampler(
    mask: np.ndarray | None,
    voxel_size: tuple[float, float, float],
    radius_multiple: float,
) -> Callable[[np.ndarray], float] | None:
    """``radius(position_um)``: the mask's distance to background, in microns,
    at the voxel nearest a physical ``(z, y, x)`` position -- or ``None`` when
    there is no mask to read, or nothing will read it.

    Built on the mask's surface voxels rather than a whole-volume distance
    transform, so it costs a KD-tree over the surface, not ~70 bytes a voxel.
    """
    if mask is None or radius_multiple <= 0:
        return None
    from haemolynx.preprocessing.pointwise_distance import FeatureDistance

    binary = np.asanyarray(mask, dtype=bool)
    distance = FeatureDistance(binary, feature_value=False, sampling=voxel_size)
    if not distance.has_surface:
        return None
    spacing = np.asarray(voxel_size, dtype=float)
    upper = np.asarray(binary.shape) - 1

    def radius(position_um: np.ndarray) -> float:
        index = np.clip(np.rint(np.asarray(position_um, dtype=float) / spacing), 0, upper)
        return float(distance.at(index.astype(np.intp).reshape(1, -1))[0])

    # Shared with the mask-support checks, so one mask's surface tree is built once.
    radius.feature_distance = distance
    return radius


def mask_continues_past(support: MaskSupport) -> Callable[[np.ndarray, np.ndarray], bool]:
    """``continues(tip_um, outward)``: whether the mask runs on past a stub's
    tip -- one and two voxels of the finest axis beyond the tip's own lumen
    radius, along *outward*, both in the mask. A blind end does not."""
    finest = float(min(support.voxel_size_zyx))

    def continues(tip_um: np.ndarray, outward: np.ndarray) -> bool:
        tip = np.asarray(tip_um, dtype=float).reshape(1, 3)
        reach = float(support.radius(tip)[0]) + finest * np.array([1.0, 2.0])
        return bool(np.all(support.inside(tip + reach[:, None] * np.asarray(outward, dtype=float))))

    return continues


def _scalar_radius(support: MaskSupport) -> Callable[[np.ndarray], float]:
    return lambda position_um: float(support.radius(np.asarray(position_um, dtype=float).reshape(1, 3))[0])


def mask_lumen_test(
    mask: np.ndarray | None,
    voxel_size: tuple[float, float, float],
) -> Callable[[np.ndarray], np.ndarray] | None:
    """``inside(points_um)``: whether each physical ``(z, y, x)`` point falls in
    the mask's lumen (a True voxel; outside the volume is background) -- or
    ``None`` when there is no mask to read.

    What degree-2 merging and direction-aware collapse ask when two edges
    join the same two nodes: one vessel, or two with background between
    (``_helpers.paths_separated_by_background``).
    """
    if mask is None:
        return None
    binary = np.asanyarray(mask, dtype=bool)

    def inside(points_um: np.ndarray) -> np.ndarray:
        return points_inside_mask(points_um, binary, voxel_size_zyx=voxel_size)

    return inside


def _log_degree2_diagnostics(G: nx.MultiGraph, max_degree: int, debug: bool) -> None:
    if not debug:
        return
    degree2_diag = diagnose_degree2_nodes(G, max_degree=max_degree)
    logger.debug(format_degree2_diagnostics_report(degree2_diag))


def build_graph_from_skeleton(
    skeleton: np.ndarray,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    graph_reconnect_threshold: float = 10.0,
    final_orphan_reconnect_threshold: float = 3.0,
    cluster_collapse_distance: float = 5.0,
    min_stub_length: float = 10.0,
    debug: bool = False,
    step_callback: StepCallback | None = None,
    cluster_collapse_method: str = "distance_only",
    cluster_collapse_max_radial_dispersion: float = DEFAULT_MAX_RADIAL_DISPERSION,
    cluster_collapse_persistence_search_multiple: float = DEFAULT_SEARCH_RADIUS_MULTIPLE,
    cluster_collapse_direction_aware_min_degree: int = DEFAULT_MIN_DEGREE_FOR_DISPERSION_CHECK,
    cluster_collapse_direction_aware_tangent_length_um: float = DEFAULT_TANGENT_LENGTH_UM,
    use_memmap: bool = False,
    segmentation_mask: np.ndarray | None = None,
    min_stub_length_radius_multiple: float = 0.0,
    protect_image_face_stubs: bool = True,
    stub_radius_at: Callable[[np.ndarray], float] | None = None,
    gap_bridge_max_turn_deg: float | None = MAX_GAP_BRIDGE_TURN_DEG,
    bridge_require_mask_support: bool = True,
    bridge_max_background_gap_um: float = DEFAULT_MAX_BACKGROUND_GAP_UM,
    bridge_min_mask_fraction: float = DEFAULT_MIN_MASK_FRACTION,
    recover_uncovered_mask_vessels: bool = True,
    recovery_min_region_volume_um3: float = DEFAULT_MIN_REGION_VOLUME_UM3,
    recovery_min_length_um: float = DEFAULT_MIN_LENGTH_UM,
) -> nx.MultiGraph:
    """
    Build and clean a vascular NetworkX graph from a binary 3D skeleton.

    Runs the full topology pipeline: skan extraction, loop stitching, secondary
    loop reconnection, topology optimisation, degree-2 removal passes, cluster
    collapse, stub pruning, self-edge removal, and orphan reconnection.

    Parameters
    ----------
    skeleton
        Binary 3D skeleton array.
    voxel_size
        Spacing of each array axis in microns, in canonical ``(z, y, x)`` order —
        *not* the ``(x, y, z)`` order reported by image metadata. Convert with
        ``haemolynx.io.voxel_size_zyx_from_xyz`` before calling.
    graph_reconnect_threshold
        Reconnection threshold for initial graph build and topology optimisation.
    final_orphan_reconnect_threshold
        Reconnection threshold for orphan/dangling node repair.
    cluster_collapse_distance
        Distance threshold for collapsing nearby node clusters.
    min_stub_length
        Minimum stub length (microns) retained before pruning, wherever the
        stub is not judged by its parent vessel's radius instead.
    debug
        When True, print degree-2 diagnostic reports after cleanup passes.
    step_callback
        Optional ``callback(G, step_label)`` invoked after each topology step.
    cluster_collapse_method
        ``"distance_only"`` (default) collapses each node within
        ``cluster_collapse_distance`` of a representative into it, clusters
        bounded by that distance (see ``collapse.collapse_node_clusters``). ``"direction_
        aware"`` additionally refuses a merge that would turn the collapsed
        node's incident edges into a cartwheel shape -- see
        ``direction_aware_collapse`` for why and how. ``"persistence"`` cuts
        each local cluster at its own 0-dimensional-persistence gap instead
        of one global distance -- see ``persistence_collapse`` for the cited
        mathematics and why it can leave more than one representative node
        where ``distance_only`` would merge everything into one.
    cluster_collapse_max_radial_dispersion
        Only read when *cluster_collapse_method* is ``"direction_aware"`` --
        see ``direction_aware_collapse.collapse_node_clusters_direction_aware``.
    cluster_collapse_persistence_search_multiple
        Only read when *cluster_collapse_method* is ``"persistence"`` -- see
        ``persistence_collapse.collapse_node_clusters_persistence``.
    cluster_collapse_direction_aware_min_degree, cluster_collapse_direction_aware_tangent_length_um
        Only read when *cluster_collapse_method* is ``"direction_aware"`` --
        deliberately the same ``cartwheel_hub_min_degree`` /
        ``cartwheel_hub_tangent_length_um`` settings the cartwheel hub guard
        itself uses, since this collapse method gates merges with that
        guard's own geometry: tuning one without the other would silently
        decouple the diagnostic from the corrective gate it is modelled on.
    use_memmap
        The low-RAM option. Builds skan's paths from the skeleton's foreground
        coordinates instead of from whole-volume copies of it, and keeps the
        secondary-loop reconnection to bounded windows -- see
        ``build.skan_skeleton`` and ``reconnect.reconnect_secondary_loop_edges``.
        The graph is the same either way.
    segmentation_mask, min_stub_length_radius_multiple
        With the binary mask the skeleton came from and a positive multiple,
        a stub is pruned when shorter than that many radii of the vessel it
        leaves (the mask's distance to background at the junction), instead
        of *min_stub_length* -- see ``prune.prune_vascular_stubs``. Whatever
        the multiple, the mask also decides when two edges joining the same
        two nodes are one vessel: degree-2 merging and direction-aware
        collapse keep both only where background separates them
        (:func:`mask_lumen_test`), not wherever they are more than 3-5 um
        apart -- Lee thinning leaves strands 5-10 um apart through one wide
        vessel. Without a mask, distance alone decides.
    stub_radius_at
        A ready-made ``radius(position_um)`` (see :func:`mask_radius_sampler`)
        used in place of building one from *segmentation_mask* -- for a
        caller building many graphs from one mask.
    protect_image_face_stubs
        Keep a short stub whose end lies within its pruning threshold of an
        image face: a vessel cut by the image, the open ends inlets and
        outlets are chosen from.
    gap_bridge_max_turn_deg
        A straight bridge between two terminals (*graph_reconnect_threshold*,
        *final_orphan_reconnect_threshold*) is refused when it turns more than
        this from a terminal's own end direction, i.e. folds back on the
        vessel it should continue, or when a terminal ends less than
        ``build.MIN_GAP_BRIDGE_END_LENGTH_UM`` of centreline -- see
        ``build.MAX_GAP_BRIDGE_TURN_DEG``. ``None`` bridges regardless.
    bridge_require_mask_support, bridge_max_background_gap_um, bridge_min_mask_fraction
        With a *segmentation_mask* and *bridge_require_mask_support* on, the
        gap, optimise and orphan reconnects draw a bridge only along a route
        through the mask crossing at most *bridge_max_background_gap_um* of
        background in one run, with at least *bridge_min_mask_fraction* of it
        in the mask, and not beside another edge in the same lumen
        (``reconnect.MaskBridges``); the stub prunes also drop stubs mostly
        off the mask or inside their parent's lumen, and hold one whose tip
        the mask runs on past to *min_stub_length* as well
        (``prune.prune_vascular_stubs``); every loop lying inside one lumen
        -- a Lee-thinning ring, round nothing or a dropout rather than tissue
        -- loses an arc (``lumen_loops.remove_loops_inside_one_lumen``); and
        a final prune runs after the orphan reconnect and recovery, so the
        stubs they leave are judged too. Without a mask, or off, none of it
        happens and those two steps leave the graph as it is.
    recover_uncovered_mask_vessels, recovery_min_region_volume_um3, recovery_min_length_um
        With a *segmentation_mask*, trace the mask the graph does not cover
        and join it to the network through the mask -- see
        ``mask_recovery.recover_uncovered_mask_vessels``. Without a mask, or
        off, the step leaves the graph as it is.

    Returns
    -------
    nx.MultiGraph
        Cleaned vascular graph.
    """
    degree2_pass1_max_degree = 4
    degree2_pass2_max_degree = 8
    inside_lumen = mask_lumen_test(segmentation_mask, voxel_size)
    stub_radius_at = (
        stub_radius_at
        if stub_radius_at is not None
        else mask_radius_sampler(segmentation_mask, voxel_size, min_stub_length_radius_multiple)
    )
    mask_support = None
    if segmentation_mask is not None and (
        bridge_require_mask_support or recover_uncovered_mask_vessels
    ):
        mask_support = MaskSupport(
            np.asanyarray(segmentation_mask, dtype=bool),
            voxel_size,
            max_background_gap_um=bridge_max_background_gap_um,
            min_mask_fraction=bridge_min_mask_fraction,
            feature_distance=getattr(stub_radius_at, "feature_distance", None),
        )
    bridge_support = mask_support if bridge_require_mask_support else None
    stub_mask_rules = {}
    if bridge_support is not None:
        stub_mask_rules = dict(
            inside_lumen=inside_lumen,
            mask_continues_at=mask_continues_past(bridge_support),
        )
    image_extent_um = (
        (np.asarray(skeleton.shape, dtype=float) - 1.0) * np.asarray(voxel_size, dtype=float)
        if protect_image_face_stubs
        else None
    )

    def prune_stubs(G: nx.MultiGraph) -> nx.MultiGraph:
        radius_at = stub_radius_at
        if radius_at is None and bridge_support is not None:
            radius_at = _scalar_radius(bridge_support)
        return prune_vascular_stubs(
            G,
            debug=debug,
            min_stub_length=min_stub_length,
            radius_at=radius_at,
            radius_multiple=float(min_stub_length_radius_multiple),
            image_extent_um=image_extent_um,
            **stub_mask_rules,
        )

    logger.info("Building skan Skeleton object...")
    warmup = skan_numba_warmup_skeleton()
    if warmup is not None:
        csr.Skeleton(warmup)
    sk = skan_skeleton(skeleton, use_memmap=use_memmap)
    logger.info(f"skan Skeleton built: {sk.n_paths} paths")

    logger.info("Building graph (loop detection + segment extraction)...")
    G, voxel_loops, loop_edges = build_graph_segment_skan_stitched_loops(
        sk,
        skeleton,
        debug=debug,
        voxel_size=voxel_size,
        reconnect_threshold=graph_reconnect_threshold,
        max_bridge_turn_deg=gap_bridge_max_turn_deg,
        mask_support=bridge_support,
    )
    _notify_step(G, "build_graph_segment_skan_stitched_loops", step_callback)

    G = reconnect_secondary_loop_edges(
        G,
        skeleton,
        voxel_size=voxel_size,
        debug=debug,
        use_memmap=use_memmap,
    )
    _notify_step(G, "reconnect_secondary_loop_edges", step_callback)

    G, _ = optimise_graph_topology_fixed(
        G,
        voxel_loops,
        loop_edges,
        skeleton_data=skeleton,
        debug=debug,
        reconnect_threshold=graph_reconnect_threshold,
        max_bridge_turn_deg=gap_bridge_max_turn_deg,
        mask_support=bridge_support,
    )
    _notify_step(G, "optimise_graph_topology_fixed", step_callback)

    G = smart_multigraph_degree2_removal(
        G,
        skeleton,
        max_degree=degree2_pass1_max_degree,
        debug=debug,
        inside_lumen=inside_lumen,
    )
    _notify_step(G, "smart_multigraph_degree2_removal_pass1", step_callback)
    _log_degree2_diagnostics(G, degree2_pass1_max_degree, debug)

    if cluster_collapse_method == "direction_aware":
        G = collapse_node_clusters_direction_aware(
            G,
            distance_threshold=cluster_collapse_distance,
            max_radial_dispersion=cluster_collapse_max_radial_dispersion,
            min_degree_for_dispersion_check=cluster_collapse_direction_aware_min_degree,
            tangent_length_um=cluster_collapse_direction_aware_tangent_length_um,
            debug=debug,
            inside_lumen=inside_lumen,
        )
    elif cluster_collapse_method == "persistence":
        G = collapse_node_clusters_persistence(
            G,
            distance_threshold=cluster_collapse_distance,
            search_radius_multiple=cluster_collapse_persistence_search_multiple,
            debug=debug,
        )
    elif cluster_collapse_method == "distance_only":
        G = collapse_node_clusters(
            G,
            distance_threshold=cluster_collapse_distance,
            debug=debug,
        )
    else:
        raise ValueError(
            f"cluster_collapse_method must be 'distance_only', 'direction_aware' "
            f"or 'persistence', got {cluster_collapse_method!r}."
        )
    _notify_step(G, "collapse_node_clusters", step_callback)

    G = smart_multigraph_degree2_removal(
        G,
        skeleton,
        max_degree=degree2_pass2_max_degree,
        debug=debug,
        inside_lumen=inside_lumen,
    )
    _notify_step(G, "smart_multigraph_degree2_removal_post_collapse", step_callback)

    G = prune_stubs(G)
    _notify_step(G, "prune_vascular_stubs", step_callback)
    _log_degree2_diagnostics(G, degree2_pass2_max_degree, debug)

    G = smart_multigraph_degree2_removal(
        G,
        skeleton,
        max_degree=degree2_pass2_max_degree,
        debug=debug,
        inside_lumen=inside_lumen,
    )
    _notify_step(G, "smart_multigraph_degree2_removal_post_prune", step_callback)
    _log_degree2_diagnostics(G, degree2_pass2_max_degree, debug)

    G = remove_edges_for_self_connected_nodes(G)
    _notify_step(G, "remove_edges_for_self_connected_nodes", step_callback)

    G = reconnect_orphan_and_dangling_nodes(
        G,
        skeleton_data=skeleton,
        reconnect_threshold=final_orphan_reconnect_threshold,
        include_degree1=True,
        max_new_edges_per_node=1,
        validate_reconnections=True,
        debug=debug,
        # Direction-aware collapse only guards its own step (above); without
        # this, orphan/dangling reconnection -- which runs later in this
        # same pipeline with no cartwheel-awareness of its own -- can
        # independently attach many separate dangling stubs to the same
        # nearby node and recreate the wheel shape the collapse step just
        # spent its own pass preventing. Only turned on for the method that
        # makes this same trade-off everywhere else in the graph.
        direction_aware=cluster_collapse_method == "direction_aware",
        max_radial_dispersion=cluster_collapse_max_radial_dispersion,
        min_degree_for_dispersion_check=cluster_collapse_direction_aware_min_degree,
        tangent_length_um=cluster_collapse_direction_aware_tangent_length_um,
        max_bridge_turn_deg=gap_bridge_max_turn_deg,
        mask_support=bridge_support,
    )
    _notify_step(G, "reconnect_orphan_and_dangling_nodes", step_callback)

    if recover_uncovered_mask_vessels and mask_support is not None:
        G = _recover_uncovered_mask_vessels(
            G,
            mask_support,
            attach_reach_um=final_orphan_reconnect_threshold,
            min_region_volume_um3=recovery_min_region_volume_um3,
            min_length_um=recovery_min_length_um,
        )
    _notify_step(G, "recover_uncovered_mask_vessels", step_callback)

    # Lee thinning rings every tunnel through the mask, and the steps above
    # can close more loops inside one vessel: one segmented vessel is drawn
    # once, so each loop that does not run round tissue loses an arc.
    if bridge_support is not None:
        G = remove_loops_inside_one_lumen(G, bridge_support)
    _notify_step(G, "remove_loops_inside_one_lumen", step_callback)

    if bridge_support is not None:
        G = prune_stubs(G)
    _notify_step(G, "prune_vascular_stubs_final", step_callback)

    G = smart_multigraph_degree2_removal(
        G,
        skeleton,
        max_degree=degree2_pass1_max_degree,
        debug=debug,
        inside_lumen=inside_lumen,
    )
    _notify_step(G, "smart_multigraph_degree2_removal_post_orphan_reconnect", step_callback)
    _log_degree2_diagnostics(G, degree2_pass2_max_degree, debug)

    return G
