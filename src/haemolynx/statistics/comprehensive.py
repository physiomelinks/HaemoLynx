"""Combining every vessel-network statistic into one report."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import networkx as nx

from haemolynx.graph.thick_vessel_junctions import IS_ZERO_RESISTANCE, contract_bridges
from haemolynx.graph.validate import assert_no_forbidden_edge_attributes

from .bifurcation import compute_daughter_daughter_angles, compute_murray_law_compliance
from .current_flow import compute_current_flow
from .inlet_outlet_routes import (
    compute_inlet_outlet_bottlenecks,
    compute_inlet_outlet_shunts,
)
from .loops import compute_loop_hierarchy
from .occlusion import compute_occlusion_curves, compute_single_vessel_occlusion_impact
from .spectral import compute_algebraic_connectivity
from .strahler import compute_strahler_orders
from .territories import compute_perfusion_territories
from .transit_time import compute_transit_times
from .network_measures import (
    compute_betweenness,
    compute_betweenness_summary,
    compute_communities,
    compute_communities_summary,
    compute_flow_hierarchy,
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
    compute_cyclomatic_number,
    compute_degree_assortativity,
    compute_k_core_structure,
    compute_network_robustness,
    compute_rich_club_coefficient,
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
    "dead_ends",
    "network_robustness",
    "bottlenecks",
    "shunts",
    "occlusion_impact",
    "occlusion_curves",
    "current_flow",
    "perfusion_territories",
    "transit_time",
    "loop_hierarchy",
    "strahler",
    "algebraic_connectivity",
    "cyclomatic_number",
    "degree_assortativity",
    "rich_club",
    "k_core",
    "flow_hierarchy",
    "path_efficiency",
    "community",
    "betweenness",
)

#: The subset of :data:`STATISTIC_MEASURES` the GUI groups under
#: "Connectivity/Network Analysis" -- graph-theoretic measures of how the
#: network is connected/arranged (bridges and loops, degree correlation,
#: shortest-path/centrality measures), nested under the schema's own
#: ``statistics_network_analysis`` toggle. Everything in STATISTIC_MEASURES
#: but not here (basic, tortuosity, branching, tree_asymmetry,
#: fractal_dimension, vessel_density, murray_law, daughter_angles,
#: intercapillary_distance) is geometric/morphometric rather than
#: connectivity, and stays directly under "statistics".
NETWORK_ANALYSIS_MEASURES: frozenset[str] = frozenset(
    {
        "network_robustness",
        "bottlenecks",
        "shunts",
        "occlusion_impact",
        "occlusion_curves",
        "current_flow",
        "perfusion_territories",
        "transit_time",
        "loop_hierarchy",
        "strahler",
        "algebraic_connectivity",
        "cyclomatic_number",
        "degree_assortativity",
        "rich_club",
        "k_core",
        "flow_hierarchy",
        "path_efficiency",
        "community",
        "betweenness",
    }
)


#: The section of :func:`compute_vessel_statistics_by_measure` holding the
#: zero-resistance bridges, which no measure counts as vessels.
ZERO_RESISTANCE_BRIDGES = "zero_resistance_bridges"


def _as_contracted(nodes: Any, became: dict) -> Any:
    """Boundary nodes as the graph with its bridges contracted names them."""
    if nodes is None:
        return None
    return list(dict.fromkeys(became.get(n, n) for n in nodes))


def flatten_statistics_sections(
    sections: Dict[str, Dict[str, Any]], statistics_mode: str
) -> Dict[str, Any]:
    """*sections* as :func:`compute_comprehensive_vessel_statistics` returns
    them: one dict, each section's rows in turn, then ``Statistics Mode``."""
    flat: Dict[str, Any] = {}
    for rows in sections.values():
        flat.update(rows)
    flat["Statistics Mode"] = statistics_mode
    return flat


def compute_comprehensive_vessel_statistics(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict] = None,
    voxel_size=(1.0, 1.0, 1.0),
    image_dimensions=None,
    statistics_mode: str = "fast",
    enabled_measures: Optional[frozenset] = None,
    inlet_nodes: Optional[Any] = None,
    outlet_nodes: Optional[Any] = None,
    route_weighting: str = "length",
    shunt_max_route_fraction: float = 0.5,
    occlusion_hypoperfusion_fraction: float = 0.5,
    occlusion_curve_max_fraction: float = 0.5,
    topology_communities: "tuple[list, str] | None" = None,
    tissue_volume_um3: Optional[float] = None,
    vessel_volume_um3: Optional[float] = None,
) -> Dict[str, Any]:
    """Combine all vessel statistics.

    Length-based metrics read the ``length`` edge attribute (microns). They never
    read ``resistance``/``conductance``, so running haemodynamics first cannot
    change a reported length.

    *enabled_measures*, when given, names the subset of :data:`STATISTIC_MEASURES`
    to actually compute -- everything else is skipped. ``None`` (the default)
    means every measure runs, matching this function's behaviour before this
    parameter existed.

    *inlet_nodes*/*outlet_nodes*, when both given, let "network_robustness"
    additionally classify which bridges/articulation points are perfusion-
    critical (see :func:`~haemolynx.statistics.topology.compute_network_robustness`).
    Omitted (the default), that measure reports exactly what it always has.

    "bottlenecks", "shunts", "occlusion_impact", "current_flow",
    "perfusion_territories", "transit_time", "loop_hierarchy", "strahler" and
    "algebraic_connectivity" also write their per-vessel results onto *G*'s
    edges so the vessels layer can be coloured by them; the route/flow ones
    measure distance and conductance by *route_weighting*. Without what they
    need (inlets and outlets, or a solved flow for transit time) they report
    N/A and write nothing.

    *topology_communities*, when given, is the ``(communities, method)``
    topological partition of *G* the "community" measure would compute --
    exact greedy modularity in ``full`` mode, ``communities_for_weighting(G,
    "topology")`` in ``fast`` -- computed once by a caller that also needs it
    (the vessels-layer vascular communities) and used instead of repeating it.

    *tissue_volume_um3* and *vessel_volume_um3*, when given, are the tissue
    measured from the raw image and the segmented vessel volume inside it
    (:func:`~haemolynx.statistics.tissue_volume.measure_tissue_volume`):
    "vessel_density" then divides by the tissue rather than by the box the
    network spans, and adds the vascular volume fraction.

    The same rows, measure by measure, are
    :func:`compute_vessel_statistics_by_measure`.
    """
    sections = compute_vessel_statistics_by_measure(
        G,
        node_positions=node_positions,
        voxel_size=voxel_size,
        image_dimensions=image_dimensions,
        statistics_mode=statistics_mode,
        enabled_measures=enabled_measures,
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        route_weighting=route_weighting,
        shunt_max_route_fraction=shunt_max_route_fraction,
        occlusion_hypoperfusion_fraction=occlusion_hypoperfusion_fraction,
        occlusion_curve_max_fraction=occlusion_curve_max_fraction,
        topology_communities=topology_communities,
        tissue_volume_um3=tissue_volume_um3,
        vessel_volume_um3=vessel_volume_um3,
    )
    return flatten_statistics_sections(sections, statistics_mode)


def compute_vessel_statistics_by_measure(
    G: Union[nx.Graph, nx.MultiGraph],
    node_positions: Optional[dict] = None,
    voxel_size=(1.0, 1.0, 1.0),
    image_dimensions=None,
    statistics_mode: str = "fast",
    enabled_measures: Optional[frozenset] = None,
    inlet_nodes: Optional[Any] = None,
    outlet_nodes: Optional[Any] = None,
    route_weighting: str = "length",
    shunt_max_route_fraction: float = 0.5,
    occlusion_hypoperfusion_fraction: float = 0.5,
    occlusion_curve_max_fraction: float = 0.5,
    topology_communities: "tuple[list, str] | None" = None,
    tissue_volume_um3: Optional[float] = None,
    vessel_volume_um3: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Every vessel statistic, one section per measure that ran.

    Keyed by the measure's name in :data:`STATISTIC_MEASURES`, in the order
    they ran, each holding the rows that measure computed; the zero-resistance
    bridges, when *G* has any, are the section :data:`ZERO_RESISTANCE_BRIDGES`.
    What the panel's Statistics tab shows, a button per section. Arguments as
    :func:`compute_comprehensive_vessel_statistics`, which is these sections
    flattened (:func:`flatten_statistics_sections`).
    """
    assert_no_forbidden_edge_attributes(G, context="vessel statistics")
    # A zero-resistance bridge is not a vessel: every statistic sees a vessel
    # opening into a thick vessel meet its centreline directly (the bridge's
    # two ends one node), and the bridges are reported on their own.
    bridge_lengths = [
        float(d.get("length") or 0.0) for *_e, d in G.edges(data=True) if d.get(IS_ZERO_RESISTANCE)
    ]
    bridge_rows: Dict[str, Any] = {}
    if bridge_lengths:
        G, became = contract_bridges(G)
        inlet_nodes = _as_contracted(inlet_nodes, became)
        outlet_nodes = _as_contracted(outlet_nodes, became)
        bridge_rows = {
            "Zero-resistance Bridges (not counted as vessels)": len(bridge_lengths),
            "Zero-resistance Bridge Length (microns)": sum(bridge_lengths),
        }
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

    sections: Dict[str, Dict[str, Any]] = {}
    if "basic" in enabled:
        sections["basic"] = compute_basic_statistics(G, is_mg)
    if "tortuosity" in enabled:
        sections["tortuosity"] = compute_tortuosity_measures(G, node_positions, is_mg)
    if "branching" in enabled:
        sections["branching"] = compute_branching_statistics(G_simple)
    if "tree_asymmetry" in enabled:
        sections["tree_asymmetry"] = compute_tree_asymmetry(G_simple)
    if "fractal_dimension" in enabled:
        # The original G, not G_simple: collapsing parallel edges to build
        # G_simple keeps only one edge's data per node pair, which would
        # silently drop other parallel edges' centreline points here.
        sections["fractal_dimension"] = compute_fractal_dimension(G, node_positions)
    if "vessel_density" in enabled:
        sections["vessel_density"] = compute_vessel_density(
            G,
            node_positions,
            voxel_size,
            image_dimensions,
            is_mg,
            tissue_volume_um3=tissue_volume_um3,
            vessel_volume_um3=vessel_volume_um3,
        )
    if "murray_law" in enabled:
        # The original G, not G_simple, for the same reason as fractal
        # dimension: a junction's parent/daughter diameters live on specific
        # parallel edges that G_simple would collapse away.
        sections["murray_law"] = compute_murray_law_compliance(G)
    if "daughter_angles" in enabled:
        sections["daughter_angles"] = compute_daughter_daughter_angles(G)
    if "intercapillary_distance" in enabled:
        # The original G, not G_simple: each parallel edge is its own
        # vessel with its own spacing to its neighbours.
        sections["intercapillary_distance"] = compute_intercapillary_distance(G)
    if "dead_ends" in enabled:
        from .dead_ends import compute_dead_end_statistics

        sections["dead_ends"] = compute_dead_end_statistics(G, tissue_volume_um3=tissue_volume_um3)
    if "network_robustness" in enabled:
        # The original G, not G_simple: a parallel edge between two
        # junctions is exactly the redundancy that keeps neither of that
        # pair a bridge, which G_simple's collapse would erase.
        sections["network_robustness"] = compute_network_robustness(
            G, inlet_nodes=inlet_nodes, outlet_nodes=outlet_nodes
        )
    # The original G for both: parallel vessels are real extra capacity and
    # extra routes, and the per-vessel results are written onto G's own edges.
    if "bottlenecks" in enabled:
        sections["bottlenecks"] = compute_inlet_outlet_bottlenecks(
            G,
            inlet_nodes,
            outlet_nodes,
            weighting=route_weighting,
            statistics_mode=statistics_mode,
        )
    if "shunts" in enabled:
        sections["shunts"] = compute_inlet_outlet_shunts(
            G,
            inlet_nodes,
            outlet_nodes,
            weighting=route_weighting,
            max_route_fraction=shunt_max_route_fraction,
        )
    if "occlusion_impact" in enabled:
        sections["occlusion_impact"] = compute_single_vessel_occlusion_impact(
            G,
            inlet_nodes,
            outlet_nodes,
            weighting=route_weighting,
            statistics_mode=statistics_mode,
            hypoperfusion_fraction=occlusion_hypoperfusion_fraction,
        )
    if "occlusion_curves" in enabled:
        sections["occlusion_curves"] = compute_occlusion_curves(
            G,
            inlet_nodes,
            outlet_nodes,
            weighting=route_weighting,
            statistics_mode=statistics_mode,
            max_fraction=occlusion_curve_max_fraction,
        )
    if "current_flow" in enabled:
        sections["current_flow"] = compute_current_flow(
            G, inlet_nodes, outlet_nodes,
            weighting=route_weighting, statistics_mode=statistics_mode,
        )
    if "perfusion_territories" in enabled:
        sections["perfusion_territories"] = compute_perfusion_territories(
            G, inlet_nodes, outlet_nodes, weighting=route_weighting
        )
    if "transit_time" in enabled:
        sections["transit_time"] = compute_transit_times(G, inlet_nodes, outlet_nodes)
    if "loop_hierarchy" in enabled:
        sections["loop_hierarchy"] = compute_loop_hierarchy(
            G,
            statistics_mode=statistics_mode,
            image_dimensions=image_dimensions,
            voxel_size=voxel_size,
        )
    if "strahler" in enabled:
        sections["strahler"] = compute_strahler_orders(G, inlet_nodes, outlet_nodes)
    if "algebraic_connectivity" in enabled:
        sections["algebraic_connectivity"] = compute_algebraic_connectivity(
            G, weighting=route_weighting
        )
    if "cyclomatic_number" in enabled:
        # The original G: a parallel edge is itself an independent loop
        # that collapsing to G_simple would erase, same reasoning as
        # network_robustness above.
        sections["cyclomatic_number"] = compute_cyclomatic_number(G)
    if "degree_assortativity" in enabled:
        sections["degree_assortativity"] = compute_degree_assortativity(G)
    if "rich_club" in enabled:
        sections["rich_club"] = compute_rich_club_coefficient(G)
    if "k_core" in enabled:
        sections["k_core"] = compute_k_core_structure(G)
    if "flow_hierarchy" in enabled:
        sections["flow_hierarchy"] = compute_flow_hierarchy(G)

    # Where their rows always were in the flattened report.
    if bridge_rows:
        sections[ZERO_RESISTANCE_BRIDGES] = bridge_rows
    if statistics_mode == "full":
        if "path_efficiency" in enabled:
            sections["path_efficiency"] = compute_path_efficiency(G, is_mg, max_pairs=None)
        if "community" in enabled:
            sections["community"] = {
                "communities": (
                    list(topology_communities[0])
                    if topology_communities is not None
                    else compute_communities(G_simple)
                )
            }
        if "betweenness" in enabled:
            sections["betweenness"] = compute_betweenness(G_simple)
        return sections

    if "path_efficiency" in enabled:
        sections["path_efficiency"] = compute_path_efficiency(G, is_mg)
    if "community" in enabled:
        sections["community"] = compute_communities_summary(
            G_simple, precomputed=topology_communities
        )
    if "betweenness" in enabled:
        sections["betweenness"] = compute_betweenness_summary(G_simple)
    return sections
