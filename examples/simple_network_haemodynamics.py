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

Boundary nodes are chosen with ``graph.select_boundary_nodes_by_method``; step
4 of ``main`` lists the other selection methods and the keyword each one reads.

Run::

    python examples/simple_network_haemodynamics.py [--output-dir DIR]
"""
import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ImageLynx import graph as graph_tools
from ImageLynx import haemodynamics, visualization
from ImageLynx.haemodynamics.poiseuille import CAPILLARY_REGIME_MAX_DIAMETER_UM


# Vessel diameters (um) per branch-order label. Art*/Ven* are above the 7 um
# capillary limit and so use the constant large-vessel viscosity.
DIAMETER_BY_BRANCH_ORDER = {"Art1": 20.0, "B01": 5.0, "Ven1": 30.0}

# Where the boundaries are, in (z, y, x) um. The nearest terminal node to each
# point is used, so these need only be approximate.
INLET_COORDINATE_ZYX = (0.0, 0.0, 0.0)
OUTLET_COORDINATE_ZYX = (0.0, 0.0, 600.0)

# Dirichlet pressures imposed there (Pa). ~45 mmHg in, ~7.5 mmHg out.
INLET_PRESSURE_PA = 6000.0
OUTLET_PRESSURE_PA = 1000.0

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


def main(output_dir: Path | str = root_dir / "examples" / "outputs" / "simple_network") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Graph.
    G = build_example_network()
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} vessels")

    # 2. Poiseuille resistance/conductance on every edge, from its diameter,
    #    length and the diameter-dependent apparent viscosity.
    G, _ = haemodynamics.apply_poiseuille_haemodynamics(
        G,
        diameter_by_branch_order=DIAMETER_BY_BRANCH_ORDER,
    )

    # 3. VTK export. Written before the flow solve because the solver reads the
    #    vessel file back to attach pressures and flows to it.
    vtk_export = visualization.graph_to_vtk(G, output_dir / "simple_network")

    # 4. Boundary nodes. Only degree-1 terminals are considered: pinning an
    #    interior junction would make it inject or remove flow mid-network.
    #    `method` selects how the terminals are picked, and each method reads a
    #    different keyword:
    #      "coordinates"           -> `coordinates`: each point snaps to the
    #                                 nearest terminal, so it need not be exact
    #      "volume"                -> `volume_boxes`: every terminal inside
    #                                 ((z,y,x), (z,y,x)) corner pairs
    #      "edge_percent"          -> `edge_percent`/`end_percent`/`axis`: the
    #                                 first/last N% along one axis; the imaging
    #                                 pipeline's default
    #      "all_degree_1"          -> every terminal in the graph
    #      "degree_1_from_starting"-> `starting_nodes_for_distance` +
    #                                 `distance_from_starting_node`
    #    With segmented masks, use graph.select_terminal_nodes_from_large_vessel_masks
    #    instead, which assigns terminals from anatomy rather than geometry.
    #    `image_shape` is only read by "edge_percent"; a real run passes the
    #    loaded image's shape, so this hand-built network synthesises one.
    #    `starting_nodes`/`output_nodes` are lists, so several inlets or outlets
    #    are simply named together and share that list's pressure.
    positions = np.asarray([G.nodes[n]["pos"] for n in G.nodes], dtype=float)
    image_shape = tuple(int(np.ceil(hi)) + 1 for hi in positions.max(axis=0))
    inlet_nodes = graph_tools.select_boundary_nodes_by_method(
        G,
        image_shape,
        method="coordinates",
        node_role="input",
        coordinates=[INLET_COORDINATE_ZYX],
    )
    outlet_nodes = graph_tools.select_boundary_nodes_by_method(
        G,
        image_shape,
        method="coordinates",
        node_role="output",
        coordinates=[OUTLET_COORDINATE_ZYX],
        exclude_nodes=inlet_nodes,
    )
    print(f"Boundary nodes: inlets={inlet_nodes}, outlets={outlet_nodes}")

    conductance, node_list = haemodynamics.build_conductance_matrix_from_graph(G)
    flow_result, vtk_export = haemodynamics.solve_flow_from_conductance_matrix(
        conductance,
        node_list,
        input_p_bc=INLET_PRESSURE_PA,
        output_p_bc=OUTLET_PRESSURE_PA,
        starting_nodes=inlet_nodes,
        output_nodes=outlet_nodes,
        vtk_export=vtk_export,
    )

    # 5. Network-level check: the flow driven through the inlet must match the
    #    pressure drop divided by the inlet-to-outlet effective resistance.
    laplacian = haemodynamics.calc_laplacian_from_conductance_matrix(conductance)
    effective_resistance = haemodynamics.calc_two_point_from_laplacian_matrix_nodeID(
        laplacian, G, inlet_nodes[0], outlet_nodes[0]
    )
    pressure = flow_result["pressure"]
    inlet_idx = node_list.index(inlet_nodes[0])
    inlet_flow = float(np.sum(conductance[inlet_idx, :] * (pressure[inlet_idx] - pressure)))

    print(f"\nViscosity regime split at {CAPILLARY_REGIME_MAX_DIAMETER_UM} um:")
    for branch_order, diameter in sorted(DIAMETER_BY_BRANCH_ORDER.items()):
        viscosity = haemodynamics.PoiseuilleModel.calculate_viscosity(diameter)
        regime = "capillary law" if diameter <= CAPILLARY_REGIME_MAX_DIAMETER_UM else "large-vessel constant"
        print(f"  {branch_order:5s} d={diameter:5.1f} um  mu={viscosity * 1e3:.2f} mPa.s  ({regime})")

    print(f"\nPressure drop:         {INLET_PRESSURE_PA - OUTLET_PRESSURE_PA:.0f} Pa")
    print(f"Effective resistance:  {effective_resistance:.3e} Pa.s/m^3")
    print(f"Inlet flow:            {inlet_flow * M3_PER_S_TO_NL_PER_MIN:.3f} nL/min")
    print(f"  from dP/R_eff:       "
          f"{(INLET_PRESSURE_PA - OUTLET_PRESSURE_PA) / effective_resistance * M3_PER_S_TO_NL_PER_MIN:.3f} nL/min")
    print(f"\nVTK with pressures and flows: {vtk_export['vessels_flow_path']}")

    return {
        "graph": G,
        "flow_result": flow_result,
        "vtk_export": vtk_export,
        "effective_resistance": effective_resistance,
        "inlet_nodes": inlet_nodes,
        "outlet_nodes": outlet_nodes,
        "inlet_flow_m3_s": inlet_flow,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "examples" / "outputs" / "simple_network",
        help="Directory for the VTK output files.",
    )
    args = parser.parse_args()
    main(args.output_dir)
