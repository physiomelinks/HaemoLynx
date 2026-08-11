#!/usr/bin/env python3
"""Minimal end-to-end haemodynamics on a hand-built vessel network.

Graph -> boundary conditions -> conductance matrix -> flow solve -> VTK.

Everything upstream of the graph (segmentation, skeletonisation, graph
assembly) is deliberately skipped so the haemodynamics API is visible on its
own; the network below is written out by hand. For the imaging pipeline see
``examples/resistance_network_pipeline.py``.

The network spans both viscosity regimes: 5 um capillaries use the calibrated
capillary power law, while the 20 um arteriole and 30 um venule sit above the
7 um limit and take the constant large-vessel viscosity (see issue #90).

Every setting lives in ``simple_network_config.yaml``, described by
``simple_network_schema.py``. Change a value there rather than editing this
script. The command line is generated from the schema, so every setting has a
flag of its own, and ``--list-settings`` and ``--save-config`` come for free::

    python examples/simple_network_haemodynamics.py
    python examples/simple_network_haemodynamics.py --config my_config.yaml
    python examples/simple_network_haemodynamics.py --inlet-pressure-pa 8000
    python examples/simple_network_haemodynamics.py --list-settings
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
src_dir = root_dir / "src"
examples_dir = root_dir / "examples"
for _path in (src_dir, examples_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from haemolynx import graph as graph_tools
from haemolynx import haemodynamics, visualization
from haemolynx.parsers import configure_console_logging, settings_from_command_line
from simple_network_schema import SCHEMA

CONFIG_PATH = examples_dir / "simple_network_config.yaml"

# Unit conversion, not a setting: nothing about a run should change it.
M3_PER_S_TO_NL_PER_MIN = 6.0e13


def build_example_network() -> nx.MultiGraph:
    """One arteriole feeding two parallel capillary paths, draining to a venule.

    Node positions are in um. Each edge carries the attributes haemodynamics
    needs (``length``, ``branch_order``) plus the ``voxels`` polyline that the
    VTK export draws.

        0 --Art1-- 1 --B01-- 2 --B01-- 4 --B01-- 6 --Ven1-- 7
                     \\               |               /
                      3 ----B01----- 5 ---B01--------
    """
    positions = {
        0: (0.0, 0.0, 0.0),
        1: (0.0, 0.0, 100.0),
        2: (-40.0, 0.0, 200.0),
        3: (40.0, 0.0, 200.0),
        4: (-40.0, 0.0, 400.0),
        5: (40.0, 0.0, 400.0),
        6: (0.0, 0.0, 500.0),
        7: (0.0, 0.0, 600.0),
    }
    vessels = [
        (0, 1, "Art1"),  # feeding arteriole
        (1, 2, "B01"),   # capillary bed: two parallel paths...
        (1, 3, "B01"),
        (2, 4, "B01"),
        (3, 5, "B01"),
        (4, 5, "B01"),   # ...cross-connected, so the solve is not just series/parallel
        (4, 6, "B01"),
        (5, 6, "B01"),
        (6, 7, "Ven1"),  # draining venule
    ]

    G = nx.MultiGraph()
    for node_id, pos in positions.items():
        G.add_node(node_id, pos=np.asarray(pos, dtype=float))

    for u, v, branch_order in vessels:
        start = G.nodes[u]["pos"]
        end = G.nodes[v]["pos"]
        G.add_edge(
            u,
            v,
            length=float(np.linalg.norm(end - start)),
            branch_order=branch_order,
            voxels=[start.tolist(), end.tolist()],
        )
    return G


def main(settings: dict) -> dict:
    """Run the pipeline for one settings dict, as loaded from the config file.

    This example has 6 settings, so each stage below is given the individual
    entries it needs rather than the whole dict -- the project convention is to
    pass the dict only once a call would otherwise take more than
    ``DICT_ARGUMENT_THRESHOLD`` (6) of them, as the imaging pipeline does. Doing
    it this way keeps each call's real dependencies visible.
    """
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Graph.
    G = build_example_network()
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} vessels")

    # 2. Poiseuille resistance/conductance on every edge, from its diameter,
    #    length and the diameter-dependent apparent viscosity.
    G, _ = haemodynamics.apply_poiseuille_haemodynamics(
        G,
        diameter_by_branch_order=settings["diameter_by_branch_order"],
    )

    # 4. Boundary nodes. Only degree-1 terminals are considered: pinning an
    #    interior junction would make it inject or remove flow mid-network.
    #    `method` selects how the terminals are picked, and each method reads a
    #    different keyword:
    #      "coordinates"           -> `coordinates`: each point snaps to the
    #                                 nearest terminal, so it need not be exact
    #      "volume"                -> `volume_boxes`: every terminal inside
    #                                 ((z,y,x), (z,y,x)) corner pairs
    #      "edge_percent"          -> `edge_percent`/`end_percent`/`axis`: the
    #                                 first/last N% of the network along one
    #                                 axis; the imaging pipeline's default,
    #                                 since it needs nothing from the dataset
    #      "all_degree_1"          -> every terminal in the graph
    #      "degree_1_from_starting"-> `starting_nodes_for_distance` +
    #                                 `distance_from_starting_node`
    #    With segmented masks, use graph.select_terminal_nodes_from_large_vessel_masks
    #    instead, which assigns terminals from anatomy rather than geometry.
    #    `image_shape` is what "edge_percent" checks its `axis` against; the
    #    bands themselves span the network, so a real run passes the loaded
    #    image's shape and this hand-built network synthesises one.
    #    `starting_nodes`/`output_nodes` are lists, so several inlets or outlets
    #    are simply named together and share that list's pressure.
    positions = np.asarray([G.nodes[n]["pos"] for n in G.nodes], dtype=float)
    image_shape = tuple(int(np.ceil(hi)) + 1 for hi in positions.max(axis=0))
    inlet_nodes = graph_tools.select_boundary_nodes_by_method(
        G,
        image_shape,
        method="coordinates",
        node_role="input",
        coordinates=[settings["inlet_coordinate_zyx"]],
    )
    outlet_nodes = graph_tools.select_boundary_nodes_by_method(
        G,
        image_shape,
        method="coordinates",
        node_role="output",
        coordinates=[settings["outlet_coordinate_zyx"]],
        exclude_nodes=inlet_nodes,
    )
    print(f"Boundary nodes: inlets={inlet_nodes}, outlets={outlet_nodes}")

    conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
    flow_result = haemodynamics.solve_flow_from_conductance_matrix(
        conductance,
        node_list,
        input_p_bc=settings["inlet_pressure_pa"],
        output_p_bc=settings["outlet_pressure_pa"],
        starting_nodes=inlet_nodes,
        output_nodes=outlet_nodes,
    )
    # Flows live on the graph, so the export below writes them like any other
    # edge attribute.
    haemodynamics.set_edge_flows(G, node_list, flow_result["pressure"])

    # 5. Network-level check: the flow driven through the inlet must match the
    #    pressure drop divided by the inlet-to-outlet effective resistance.
    laplacian = haemodynamics.calc_laplacian_from_conductance_matrix(conductance)
    effective_resistance = haemodynamics.calc_two_point_from_laplacian_matrix_nodeID(
        laplacian, G, inlet_nodes[0], outlet_nodes[0]
    )
    pressure = flow_result["pressure"]
    inlet_idx = node_list.index(inlet_nodes[0])
    inlet_flow = float(np.sum(conductance[inlet_idx, :] * (pressure[inlet_idx] - pressure)))

    # These defaults have to be the model's own, or the panel below reports a
    # law the resistances were not computed with.
    law = settings.get("viscosity_law", "pries")
    haematocrit = settings.get("haematocrit", 0.45)
    basis = settings.get("diameter_basis", "plasma_column")
    print(f"\nViscosity by {haemodynamics.describe_law(law, haematocrit, basis)}:")
    for branch_order, diameter in sorted(settings["diameter_by_branch_order"].items()):
        viscosity = haemodynamics.viscosity_for(
            diameter, law=law, haematocrit=haematocrit, diameter_basis=basis
        )
        low, high = haemodynamics.validity_range_um(law)
        regime = "fitted" if low <= diameter <= high else "extrapolated"
        print(f"  {branch_order:5s} d={diameter:5.1f} um  mu={viscosity * 1e3:.2f} mPa.s  ({regime})")

    pressure_drop = settings["inlet_pressure_pa"] - settings["outlet_pressure_pa"]
    print(f"\nPressure drop:         {pressure_drop:.0f} Pa")
    print(f"Effective resistance:  {effective_resistance:.3e} Pa.s/m^3")
    print(f"Inlet flow:            {inlet_flow * M3_PER_S_TO_NL_PER_MIN:.3f} nL/min")
    print(f"  from dP/R_eff:       "
          f"{pressure_drop / effective_resistance * M3_PER_S_TO_NL_PER_MIN:.3f} nL/min")
    # 6. Export, once, after everything that writes to the graph.
    vtk_export = visualization.graph_to_vtk(G, output_dir / "simple_network")
    print(f"\nVTK with pressures and flows: {vtk_export['vessels_path']}")

    return {
        "graph": G,
        "flow_result": flow_result,
        "vtk_export": vtk_export,
        "effective_resistance": effective_resistance,
        "inlet_nodes": inlet_nodes,
        "outlet_nodes": outlet_nodes,
        "inlet_flow_m3_s": inlet_flow,
        "settings": settings,
    }


if __name__ == "__main__":
    # `haemodynamics` logs what it works out about each vessel; this script's
    # own results below are printed. Both go to stdout, in order.
    configure_console_logging()
    main(settings_from_command_line(SCHEMA, CONFIG_PATH, description=__doc__))
