"""Combining every vessel-network statistic into one report."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import networkx as nx

from haemolynx.graph.validate import assert_no_forbidden_edge_attributes

from .bifurcation import compute_daughter_daughter_angles, compute_murray_law_compliance
from .network_measures import (
    compute_betweenness,
    compute_betweenness_summary,
    compute_communities,
    compute_communities_summary,
)
from .shape import (
    compute_fractal_dimension,
    compute_intercapillary_distance,
    compute_path_efficiency,
    compute_tortuosity_measures,
    compute_vessel_density,
)
from .topology import (
    compute_basic_statistics,
    compute_branching_statistics,
    compute_network_robustness,
    compute_tree_asymmetry,
)

#: Every independently-toggleable group compute_comprehensive_vessel_statistics
#: can compute, in the order it evaluates them. GUI/schema settings name a
#: measure by one of these strings (see pipeline.schema's statistics_*
#: settings); a caller passing its own enabled_measures must use these names.
STATISTIC_MEASURES: tuple[str, ...] = (
    "basic",
    "tortuosity",
    "branching",
    "tree_asymmetry",
    "fractal_dimension",
    "vessel_density",
    "murray_law",
    "daughter_angles",
    "intercapillary_distance",
    "network_robustness",
    "path_efficiency",
    "community",
    "betweenness",
)


def compute_comprehensive_vessel_statistics(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict] = None,
    voxel_size=(1.0, 1.0, 1.0),
    image_dimensions=None,
    statistics_mode: str = "fast",
    enabled_measures: Optional[frozenset] = None,
) -> Dict[str, Any]:
    """Combine all vessel statistics.

    Length-based metrics read the ``length`` edge attribute (microns). They never
    read ``resistance``/``conductance``, so running haemodynamics first cannot
    change a reported length.

    *enabled_measures*, when given, names the subset of :data:`STATISTIC_MEASURES`
    to actually compute -- everything else is skipped. ``None`` (the default)
    means every measure runs, matching this function's behaviour before this
    parameter existed.
    """
    assert_no_forbidden_edge_attributes(G, context="vessel statistics")
    valid_modes = {"fast", "full"}
    if statistics_mode not in valid_modes:
        raise ValueError(
            f"Invalid statistics_mode='{statistics_mode}'. "
            f"Choose one of {sorted(valid_modes)}."
        )

    all_measures = frozenset(STATISTIC_MEASURES)
    enabled = all_measures if enabled_measures is None else frozenset(enabled_measures)
    unknown = enabled - all_measures
    if unknown:
        raise ValueError(
            f"Unknown statistic measure(s): {sorted(unknown)}. "
            f"Choose from {STATISTIC_MEASURES}."
        )

    is_mg = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    G_simple = (
        (nx.Graph(G) if not G.is_directed() else nx.DiGraph(G))
        if is_mg
        else G
    )

    base: Dict[str, Any] = {}
    if "basic" in enabled:
        base.update(compute_basic_statistics(G, is_mg))
    if "tortuosity" in enabled:
        base.update(compute_tortuosity_measures(G, node_positions, is_mg))
    if "branching" in enabled:
        base.update(compute_branching_statistics(G_simple))
    if "tree_asymmetry" in enabled:
        base.update(compute_tree_asymmetry(G_simple))
    if "fractal_dimension" in enabled:
        # The original G, not G_simple: collapsing parallel edges to build
        # G_simple keeps only one edge's data per node pair, which would
        # silently drop other parallel edges' centreline points here.
        base.update(compute_fractal_dimension(G, node_positions))
    if "vessel_density" in enabled:
        base.update(
            compute_vessel_density(G, node_positions, voxel_size, image_dimensions, is_mg)
        )
    if "murray_law" in enabled:
        # The original G, not G_simple, for the same reason as fractal
        # dimension: a junction's parent/daughter diameters live on specific
        # parallel edges that G_simple would collapse away.
        base.update(compute_murray_law_compliance(G))
    if "daughter_angles" in enabled:
        base.update(compute_daughter_daughter_angles(G))
    if "intercapillary_distance" in enabled:
        # The original G, not G_simple: each parallel edge is its own
        # vessel with its own spacing to its neighbours.
        base.update(compute_intercapillary_distance(G))
    if "network_robustness" in enabled:
        # The original G, not G_simple: a parallel edge between two
        # junctions is exactly the redundancy that keeps neither of that
        # pair a bridge, which G_simple's collapse would erase.
        base.update(compute_network_robustness(G))

    if statistics_mode == "full":
        if "path_efficiency" in enabled:
            base.update(compute_path_efficiency(G, is_mg, max_pairs=None))
        if "community" in enabled:
            base["communities"] = compute_communities(G_simple)
        if "betweenness" in enabled:
            base.update(compute_betweenness(G_simple))
        base["Statistics Mode"] = "full"
        return base

    if "path_efficiency" in enabled:
        base.update(compute_path_efficiency(G, is_mg))
    if "community" in enabled:
        base.update(compute_communities_summary(G_simple))
    if "betweenness" in enabled:
        base.update(compute_betweenness_summary(G_simple))
    base["Statistics Mode"] = "fast"
    return base
